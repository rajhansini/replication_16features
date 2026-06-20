import time
from functools import lru_cache
from typing import Dict, Hashable, Iterable, Tuple

from ..models.belief import InferenceResult

EvidenceKey = Tuple[Tuple[Hashable, Hashable], ...]

class InferenceCache:
    """
    Thin memoizer over propagate_all_marginals(infer, I, G, evidence).
    Use get(evidence_dict) to retrieve posteriors as a nested dict.
    """
    def __init__(self, infer, individuals, gen_states, propagate_fn, *, tuple_mode=False):
        self.infer = infer
        self.I = individuals
        self.G = gen_states
        self._propagate = propagate_fn  # callable(infer, I, G, evidence_dict) -> InferenceResult
        self.total_inference_time = 0
        self.inference_calls = 0
        self.tuple_mode = tuple_mode
        self.tuple_posteriors = {}
        
        # Cache diagnostics
        self.cache_hits = 0
        self.cache_misses = 0
        self.precomputed_keys = set()  # Track what was pre-populated from Exact DP
        self.requested_keys = set()    # Track what ADP actually requests
        self.hit_keys = set()          # Track successful cache hits
        self.miss_keys = set()         # Track cache misses
        self.diagnostic_mode = False   # Enable detailed logging

    def _key(self, evidence: Dict[Hashable, Hashable]) -> EvidenceKey:
        # stable, hashable key: sorted tuple of (i, g)
        # Enhanced canonicalization to fix potential key matching issues
        return tuple(sorted((k, v) for k, v in evidence.items() if v is not None))

    @lru_cache(maxsize=200_000)
    def posteriors(self, evidence_key: EvidenceKey):
        evidence = dict(evidence_key)
        start_time = time.time()
        result = self._propagate(self.infer, self.I, self.G, evidence)
        end_time = time.time()
        self.total_inference_time += (end_time - start_time)
        self.inference_calls += 1
        return result

    def get(self, evidence: Dict[Hashable, Hashable]):
        evidence_key = self._key(evidence)
        self.requested_keys.add(evidence_key)
        
        # Check if this will be a cache hit or miss
        cache_info = self.posteriors.cache_info()
        hits_before = cache_info.hits
        
        result = self.posteriors(evidence_key)
        
        # Check if we got a cache hit
        cache_info_after = self.posteriors.cache_info()
        if cache_info_after.hits > hits_before:
            # Cache hit
            self.cache_hits += 1
            self.hit_keys.add(evidence_key)
            if self.diagnostic_mode:
                print(f"🎯 CACHE HIT: {dict(evidence_key)}")
        else:
            # Cache miss  
            self.cache_misses += 1
            self.miss_keys.add(evidence_key)
            if self.diagnostic_mode:
                was_precomputed = evidence_key in self.precomputed_keys
                print(f"❌ CACHE MISS: {dict(evidence_key)} {'(was precomputed!)' if was_precomputed else '(not precomputed)'}")
        
        if self.tuple_mode and hasattr(result, "get_tuple_pmfs"):
            state = frozenset(evidence.items())
            self.tuple_posteriors[state] = result.get_tuple_pmfs()

        return result
    
    def set_precomputed(self, evidence: Dict[Hashable, Hashable], result):
        """Directly set a cached result without calling inference."""
        evidence_key = self._key(evidence)
        self.precomputed_keys.add(evidence_key)

        if not isinstance(result, InferenceResult):
            result = InferenceResult(result)
        
        if self.diagnostic_mode:
            print(f"📥 PRECOMPUTING: {dict(evidence_key)}")
        
        # Manually populate the LRU cache by calling posteriors with a mock that returns the result
        # Store the original propagate function
        original_propagate = self._propagate
        
        # Temporarily replace with a function that returns the precomputed result
        self._propagate = lambda infer, I, G, evidence_dict: result
        
        # Call posteriors to populate cache (this won't do actual inference)
        _ = self.posteriors(evidence_key)
        
        # Restore the original propagate function
        self._propagate = original_propagate

        if self.tuple_mode and hasattr(result, "get_tuple_pmfs"):
            state = frozenset(evidence.items())
            self.tuple_posteriors[state] = result.get_tuple_pmfs()

    def enable_diagnostics(self, enabled=True):
        """Enable or disable detailed diagnostic logging."""
        self.diagnostic_mode = enabled

    def get_cache_stats(self):
        """Return comprehensive cache statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests) if total_requests > 0 else 0.0
        
        overlap = len(self.precomputed_keys & self.requested_keys)
        precomputed_unused = len(self.precomputed_keys - self.requested_keys)
        requested_not_precomputed = len(self.requested_keys - self.precomputed_keys)
        
        return {
            'total_requests': total_requests,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'precomputed_states': len(self.precomputed_keys),
            'requested_states': len(self.requested_keys),
            'state_overlap': overlap,
            'precomputed_unused': precomputed_unused,
            'requested_not_precomputed': requested_not_precomputed,
            'total_inference_time': self.total_inference_time,
            'inference_calls': self.inference_calls
        }

    def print_cache_summary(self, label=""):
        """Print a comprehensive cache effectiveness summary."""
        stats = self.get_cache_stats()
        print(f"\n📊 CACHE DIAGNOSTICS {label}")
        print(f"{'='*50}")
        print(f"Cache Requests: {stats['total_requests']} (Hits: {stats['cache_hits']}, Misses: {stats['cache_misses']})")
        print(f"Hit Rate: {stats['hit_rate']:.1%}")
        print(f"Inference Time: {stats['total_inference_time']:.3f}s ({stats['inference_calls']} calls)")
        print(f"\nState Space Analysis:")
        print(f"  Precomputed (Exact DP): {stats['precomputed_states']} states")
        print(f"  Requested (ADP):        {stats['requested_states']} states") 
        print(f"  Overlap:                {stats['state_overlap']} states")
        print(f"  Unused precomputed:     {stats['precomputed_unused']} states")
        print(f"  Not precomputed:        {stats['requested_not_precomputed']} states")
        
        if stats['precomputed_states'] > 0:
            utilization = stats['state_overlap'] / stats['precomputed_states']
            print(f"  Precomputed utilization: {utilization:.1%}")
        
        if stats['requested_states'] > 0:
            coverage = stats['state_overlap'] / stats['requested_states']
            print(f"  Request coverage:       {coverage:.1%}")
        print(f"{'='*50}")

    def print_state_details(self, max_items=10):
        """Print detailed state information for debugging."""
        print(f"\n🔍 STATE DETAILS (showing up to {max_items} each)")
        print(f"{'='*60}")
        
        print("\nPrecomputed but unused:")
        unused = list(self.precomputed_keys - self.requested_keys)[:max_items]
        for key in unused:
            print(f"  {dict(key)}")
        if len(self.precomputed_keys - self.requested_keys) > max_items:
            print(f"  ... and {len(self.precomputed_keys - self.requested_keys) - max_items} more")
            
        print("\nRequested but not precomputed:")
        missing = list(self.requested_keys - self.precomputed_keys)[:max_items]
        for key in missing:
            print(f"  {dict(key)}")
        if len(self.requested_keys - self.precomputed_keys) > max_items:
            print(f"  ... and {len(self.requested_keys - self.precomputed_keys) - max_items} more")
            
        print("\nSuccessful overlaps:")
        overlaps = list(self.precomputed_keys & self.requested_keys)[:max_items]
        for key in overlaps:
            print(f"  {dict(key)}")
        if len(self.precomputed_keys & self.requested_keys) > max_items:
            print(f"  ... and {len(self.precomputed_keys & self.requested_keys) - max_items} more")
        print(f"{'='*60}")


class CutManager:
    """
    Deduplicate candidate Bellman cuts to avoid near-identical rows.
    """
    def __init__(self):
        self.seen = set()

    def dedup_key(self,
                  state_key: Hashable,
                  i: Hashable,
                  pruned_succs: Iterable[Tuple[Hashable, float]],
                  rhs_const: float):
        # Round rhs_const to tolerate tiny numeric differences
        return (state_key, i, tuple(pruned_succs), round(float(rhs_const), 12))

    def is_new(self, key) -> bool:
        if key in self.seen:
            return False
        self.seen.add(key)
        return True
