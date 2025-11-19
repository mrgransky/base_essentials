import os
import platform
import subprocess
import psutil
import socket
from typing import Dict, Tuple, Optional
import time

def detect_environment_advanced(verbose: bool = False) -> Dict[str, any]:
    """
    Advanced environment detection with detailed system characteristics.
    Returns comprehensive environment info for optimal cache sizing.
    """
    env_info = {
        "platform": platform.system(),
        "hostname": socket.gethostname(),
        "total_memory_gb": psutil.virtual_memory().total / (1024**3),
        "available_memory_gb": psutil.virtual_memory().available / (1024**3),
        "cpu_count": os.cpu_count(),
        "is_vm": False,
        "is_hpc": False,
        "is_cloud": False,
        "is_laptop": False,
        "is_workstation": False,
        "storage_type": "unknown",
        "memory_pressure": "low"
    }
    
    # Enhanced VM detection
    vm_indicators = {
        "environment_vars": any(env in os.environ for env in [
            'SLURM_JOB_ID', 'PBS_JOBID', 'SGE_TASK_ID', 'LSB_JOBID'
        ]),
        "hostname_patterns": any(pattern in env_info["hostname"].lower() for pattern in [
            'compute', 'node', 'worker', 'vm', 'cloud', 'aws', 'gcp', 'azure'
        ]),
        "filesystem_indicators": any(os.path.exists(path) for path in [
            '/sys/hypervisor', '/proc/xen', '/sys/bus/vmbus'
        ])
    }
    
    # HPC detection (more comprehensive)
    hpc_indicators = {
        "slurm": "SLURM_JOB_ID" in os.environ,
        "pbs": "PBS_JOBID" in os.environ,
        "sge": "SGE_TASK_ID" in os.environ,
        "lsf": "LSB_JOBID" in os.environ,
        "scratch_dirs": any(os.path.exists(path) for path in [
            '/scratch', '/lustre', '/gpfs', '/work'
        ]),
        "module_system": any(os.path.exists(path) for path in [
            '/usr/share/Modules', '/opt/modules'
        ])
    }
    
    # Cloud detection
    cloud_indicators = {
        "aws": any(pattern in env_info["hostname"] for pattern in ['aws', 'ec2', 'amazon']),
        "gcp": any(pattern in env_info["hostname"] for pattern in ['gcp', 'google', 'gce']),
        "azure": any(pattern in env_info["hostname"] for pattern in ['azure', 'microsoft']),
        "metadata_endpoints": _check_cloud_metadata()
    }
    
    # Storage type detection
    env_info["storage_type"] = _detect_storage_type()
    
    # Memory pressure assessment
    memory_percent = psutil.virtual_memory().percent
    if memory_percent > 80:
        env_info["memory_pressure"] = "high"
    elif memory_percent > 60:
        env_info["memory_pressure"] = "medium"
    else:
        env_info["memory_pressure"] = "low"
    
    # System classification
    env_info["is_hpc"] = any(hpc_indicators.values())
    env_info["is_cloud"] = any(cloud_indicators.values())
    env_info["is_vm"] = any(vm_indicators.values()) or env_info["is_cloud"]
    env_info["is_laptop"] = env_info["total_memory_gb"] <= 32 and not env_info["is_hpc"] and not env_info["is_cloud"]
    env_info["is_workstation"] = env_info["total_memory_gb"] > 32 and not env_info["is_hpc"] and not env_info["is_cloud"]
    
    if verbose:
        print(f"🖥️  Environment Detection Results:")
        print(f"   Platform: {env_info['platform']}")
        print(f"   Type: {'HPC' if env_info['is_hpc'] else 'Cloud' if env_info['is_cloud'] else 'Workstation' if env_info['is_workstation'] else 'Laptop'}")
        print(f"   Memory: {env_info['available_memory_gb']:.1f}GB available / {env_info['total_memory_gb']:.1f}GB total")
        print(f"   Storage: {env_info['storage_type']}")
        print(f"   Memory pressure: {env_info['memory_pressure']}")
    
    return env_info

def _check_cloud_metadata() -> bool:
    """Quick check for cloud metadata endpoints (non-blocking)"""
    try:
        import socket
        # Try to connect to common cloud metadata IPs (very quick check)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.1)  # 100ms timeout
        result = sock.connect_ex(('169.254.169.254', 80))  # AWS/GCP metadata
        sock.close()
        return result == 0
    except:
        return False

def _detect_storage_type() -> str:
    """Detect primary storage type for I/O optimization"""
    try:
        if platform.system() == "Linux":
            # Check for NVMe
            if os.path.exists("/sys/block") and any("nvme" in d for d in os.listdir("/sys/block")):
                return "nvme"
            # Check for SSD
            try:
                with open("/sys/block/sda/queue/rotational", "r") as f:
                    if f.read().strip() == "0":
                        return "ssd"
            except:
                pass
            return "hdd"
        return "unknown"
    except:
        return "unknown"

def get_optimal_cache_size(
    dataset_size: int,
    image_estimate_mb: float = 7.0,
    target_coverage: float = None,  # Auto-determine if None
    safety_factor: float = 0.8,     # Use 80% of calculated safe memory
    min_cache_items: int = 100,
    max_cache_items: int = None,    # Auto-determine if None
    verbose: bool = True
) -> Tuple[int, Dict[str, any]]:
    """
    Calculate optimal cache size based on environment, dataset, and available resources.
    
    Args:
        dataset_size: Number of images in dataset
        image_estimate_mb: Estimated size per image in MB
        target_coverage: Target percentage of dataset to cache (auto if None)
        safety_factor: Safety margin for memory usage (0.8 = 80% of safe memory)
        min_cache_items: Minimum cache size
        max_cache_items: Maximum cache size (auto if None)
        verbose: Print detailed analysis
        
    Returns:
        Tuple of (cache_size, analysis_info)
    """
    
    # Get environment info
    env = detect_environment_advanced(verbose=verbose)
    
    # Determine strategy based on environment
    strategy = _determine_cache_strategy(env)
    
    # Calculate memory constraints
    memory_constraints = _calculate_memory_constraints(env, strategy, safety_factor)
    
    # Calculate coverage constraints
    coverage_constraints = _calculate_coverage_constraints(
        dataset_size, target_coverage, env, strategy
    )
    
    # Calculate final cache size
    cache_analysis = _finalize_cache_size(
        memory_constraints,
        coverage_constraints,
        image_estimate_mb,
        min_cache_items,
        max_cache_items,
        dataset_size
    )
    
    if verbose:
        _print_cache_analysis(cache_analysis, env, strategy)
    
    return cache_analysis["final_cache_size"], cache_analysis

def _determine_cache_strategy(env: Dict) -> Dict[str, any]:
    """Determine caching strategy based on environment"""
    
    if env["is_hpc"]:
        return {
            "name": "aggressive_hpc",
            "memory_usage_ratio": 0.4,  # Use up to 40% of available memory
            "target_coverage": 0.8,     # Aim for 80% dataset coverage
            "rationale": "HPC: Abundant memory, high-performance storage"
        }
    elif env["is_cloud"] and env["total_memory_gb"] > 60:
        return {
            "name": "cloud_large",
            "memory_usage_ratio": 0.3,
            "target_coverage": 0.6,
            "rationale": "Large cloud instance: Good memory, optimize for cost"
        }
    elif env["is_workstation"]:
        return {
            "name": "workstation",
            "memory_usage_ratio": 0.25,
            "target_coverage": 0.5,
            "rationale": "Workstation: Balance performance and stability"
        }
    elif env["storage_type"] == "nvme":
        return {
            "name": "nvme_optimized",
            "memory_usage_ratio": 0.15,  # Lower cache, fast storage compensates
            "target_coverage": 0.3,
            "rationale": "NVMe storage: Fast I/O reduces cache necessity"
        }
    elif env["is_laptop"]:
        return {
            "name": "laptop_conservative",
            "memory_usage_ratio": 0.1,
            "target_coverage": 0.2,
            "rationale": "Laptop: Conservative for user experience"
        }
    else:
        return {
            "name": "default_safe",
            "memory_usage_ratio": 0.15,
            "target_coverage": 0.4,
            "rationale": "Default: Balanced approach for unknown environment"
        }

def _calculate_memory_constraints(env: Dict, strategy: Dict, safety_factor: float) -> Dict:
    """Calculate memory-based constraints"""
    
    available_gb = env["available_memory_gb"]
    
    # Adjust for memory pressure
    if env["memory_pressure"] == "high":
        available_gb *= 0.5  # Very conservative if memory is tight
    elif env["memory_pressure"] == "medium":
        available_gb *= 0.7
    
    # Calculate usable memory for cache
    usable_memory_gb = available_gb * strategy["memory_usage_ratio"] * safety_factor
    
    # Platform-specific adjustments
    if env["is_vm"]:
        usable_memory_gb *= 0.8  # VMs can have overhead
    
    return {
        "available_gb": env["available_memory_gb"],
        "adjusted_available_gb": available_gb,
        "usable_memory_gb": usable_memory_gb,
        "strategy_ratio": strategy["memory_usage_ratio"],
        "safety_factor": safety_factor
    }

def _calculate_coverage_constraints(dataset_size: int, target_coverage: float, env: Dict, strategy: Dict) -> Dict:
    """Calculate coverage-based constraints"""
    
    if target_coverage is None:
        target_coverage = strategy["target_coverage"]
    
    # Adjust coverage based on dataset size
    if dataset_size < 1000:
        target_coverage = min(1.0, target_coverage * 1.5)  # Small datasets can be fully cached
    elif dataset_size > 100000:
        target_coverage = max(0.1, target_coverage * 0.5)  # Large datasets need lower coverage
    
    coverage_based_cache_size = int(dataset_size * target_coverage)
    
    return {
        "target_coverage": target_coverage,
        "coverage_based_size": coverage_based_cache_size,
        "dataset_size": dataset_size
    }

def _finalize_cache_size(
    memory_constraints: Dict,
    coverage_constraints: Dict,
    image_estimate_mb: float,
    min_cache_items: int,
    max_cache_items: Optional[int],
    dataset_size: int
) -> Dict:
    """Calculate final cache size with all constraints"""
    
    # Memory-based cache size
    memory_based_size = int((memory_constraints["usable_memory_gb"] * 1024) / image_estimate_mb)
    
    # Coverage-based cache size
    coverage_based_size = coverage_constraints["coverage_based_size"]
    
    # Take the minimum to respect both constraints
    constrained_size = min(memory_based_size, coverage_based_size)
    
    # Apply min/max constraints
    final_size = max(constrained_size, min_cache_items)
    
    if max_cache_items is not None:
        final_size = min(final_size, max_cache_items)
    
    # Ensure we don't exceed dataset size
    final_size = min(final_size, dataset_size)
    
    # Calculate actual coverage and memory usage
    actual_coverage = final_size / dataset_size
    actual_memory_gb = (final_size * image_estimate_mb) / 1024
    
    return {
        "final_cache_size": final_size,
        "memory_based_size": memory_based_size,
        "coverage_based_size": coverage_based_size,
        "actual_coverage": actual_coverage,
        "actual_memory_gb": actual_memory_gb,
        "limiting_factor": "memory" if memory_based_size < coverage_based_size else "coverage"
    }

def _print_cache_analysis(analysis: Dict, env: Dict, strategy: Dict):
    """Print detailed cache analysis"""
    
    print(f"\n🚀 OPTIMIZED CACHE ANALYSIS")
    print(f"=" * 50)
    print(f"Strategy: {strategy['name']} - {strategy['rationale']}")
    print(f"Dataset size: {analysis['coverage_based_size']} images")
    print(f"")
    print(f"💾 Memory Analysis:")
    print(f"   Available: {env['available_memory_gb']:.1f}GB")
    print(f"   Usable for cache: {analysis['actual_memory_gb']:.1f}GB")
    print(f"   Memory-based limit: {analysis['memory_based_size']:,} images")
    print(f"")
    print(f"📊 Coverage Analysis:")
    print(f"   Target coverage: {strategy['target_coverage']*100:.0f}%")
    print(f"   Actual coverage: {analysis['actual_coverage']*100:.1f}%")
    print(f"   Coverage-based size: {analysis['coverage_based_size']:,} images")
    print(f"")
    print(f"✅ Final Decision:")
    print(f"   Cache size: {analysis['final_cache_size']:,} images")
    print(f"   Memory usage: {analysis['actual_memory_gb']:.1f}GB")
    print(f"   Limiting factor: {analysis['limiting_factor']}")
    print(f"   Expected speedup: {_estimate_speedup(analysis['actual_coverage'])}")

def _estimate_speedup(coverage: float) -> str:
    """Estimate expected speedup based on coverage"""
    if coverage >= 0.9:
        return "80-90% (excellent)"
    elif coverage >= 0.7:
        return "60-80% (very good)"
    elif coverage >= 0.5:
        return "40-60% (good)"
    elif coverage >= 0.3:
        return "20-40% (moderate)"
    elif coverage >= 0.1:
        return "10-20% (minimal)"
    else:
        return "<10% (negligible)"

# Convenience function that replaces your current get_cache_size
def get_cache_size_optimized(
    dataset_size: int,
    image_estimate_mb: float = 7.0,
    verbose: bool = True
) -> int:
    """
    Optimized version that replaces your current get_cache_size function.
    
    Args:
        dataset_size: Number of images in your dataset
        image_estimate_mb: Estimated size per image
        verbose: Print analysis
        
    Returns:
        Optimal cache size for your environment
    """
    cache_size, analysis = get_optimal_cache_size(
        dataset_size=dataset_size,
        image_estimate_mb=image_estimate_mb,
        verbose=verbose
    )
    
    return cache_size

# Example usage and comparison
def compare_cache_strategies():
    """Compare old vs new cache sizing for different scenarios"""
    
    scenarios = [
        {"name": "Your SMU Dataset", "dataset_size": 1507, "memory_gb": 6.2},
        {"name": "Large Dataset", "dataset_size": 50000, "memory_gb": 32},
        {"name": "HPC Scenario", "dataset_size": 200000, "memory_gb": 190},
    ]
    
    print("CACHE STRATEGY COMPARISON")
    print("=" * 60)
    
    for scenario in scenarios:
        print(f"\nScenario: {scenario['name']}")
        print(f"Dataset: {scenario['dataset_size']:,} images")
        print(f"Available memory: {scenario['memory_gb']:.1f}GB")
        
        # Your current conservative approach
        old_cache = min(64, int(scenario['memory_gb'] * 0.02 * 1024 / 7))
        old_coverage = old_cache / scenario['dataset_size'] * 100
        
        # Optimized approach
        new_cache, _ = get_optimal_cache_size(
            dataset_size=scenario['dataset_size'],
            verbose=False
        )
        new_coverage = new_cache / scenario['dataset_size'] * 100
        
        print(f"   Old method: {old_cache:,} images ({old_coverage:.1f}% coverage)")
        print(f"   Optimized:  {new_cache:,} images ({new_coverage:.1f}% coverage)")
        print(f"   Improvement: {new_cache/old_cache:.1f}x larger cache")

if __name__ == "__main__":
    # Example usage
    compare_cache_strategies()