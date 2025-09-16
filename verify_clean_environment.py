#!/usr/bin/env python3
"""
Verification script for clean S2ANet training environment.
Run this after cleanup to ensure everything is properly configured.
"""

import sys
import os
from pathlib import Path

def check_imports():
    """Check that all required packages import from site-packages."""
    print("🔍 Checking package imports...")
    
    try:
        import mmdet
        import mmcv
        import mmrotate
        import torch
        
        # Check import locations
        packages = {
            'mmdet': mmdet.__file__,
            'mmcv': mmcv.__file__,
            'mmrotate': mmrotate.__file__,
            'torch': torch.__file__
        }
        
        project_root = Path.cwd()
        all_good = True
        
        for pkg_name, pkg_path in packages.items():
            is_site_packages = 'site-packages' in pkg_path
            status = "✓" if is_site_packages else "⚠️"
            
            print(f"{status} {pkg_name:12}: {pkg_path}")
            
            if not is_site_packages and pkg_name in ['mmdet', 'mmcv', 'mmrotate']:
                print(f"   WARNING: {pkg_name} not from site-packages - may cause conflicts!")
                all_good = False
        
        if all_good:
            print("\n✅ All critical packages importing from site-packages - Good!")
        else:
            print("\n⚠️  Some packages may cause import conflicts")
            
        return all_good
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure your conda environment is activated and packages are installed")
        return False

def check_config_file():
    """Check that the main config file exists."""
    print("\n🔍 Checking S2ANet configuration...")
    
    config_path = Path("configs/s2anet_r50_fpn_1x_dota12_base.py")
    
    if config_path.exists():
        print(f"✅ Found config: {config_path}")
        return True
    else:
        print(f"❌ Missing config: {config_path}")
        return False

def check_essential_directories():
    """Check that essential directories exist."""
    print("\n🔍 Checking essential directories...")
    
    essential_dirs = [
        "tools",
        "data", 
        "checkpoints",
        "DOTA_devkit",
        "work_dirs"
    ]
    
    all_good = True
    for dir_name in essential_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"✅ {dir_name}/")
        else:
            print(f"⚠️  {dir_name}/ (missing - may be optional)")
    
    return all_good

def check_no_local_packages():
    """Check that no local package directories exist that could shadow imports."""
    print("\n🔍 Checking for local package directories...")
    
    local_packages = ["mmdet", "mmcv", "mmrotate", "mmdet_legacy"]
    conflicts = []
    
    for pkg in local_packages:
        pkg_path = Path(pkg)
        disabled_path = Path(f"{pkg}_DISABLED")
        
        if pkg_path.exists():
            print(f"⚠️  Found local {pkg}/ - this may shadow pip installation!")
            conflicts.append(pkg)
        elif disabled_path.exists():
            print(f"✅ {pkg}_DISABLED/ (safely disabled)")
        else:
            print(f"✅ No local {pkg}/ directory")
    
    if conflicts:
        print(f"\n⚠️  Warning: {len(conflicts)} local package(s) found that may cause conflicts")
        print("Consider running the cleanup script to archive them")
        return False
    else:
        print("\n✅ No conflicting local packages found")
        return True

def check_training_readiness():
    """Test that training script can import successfully."""
    print("\n🔍 Testing training script import...")
    
    try:
        # Test the core imports that training needs
        import mmrotate
        from mmdet.datasets import build_dataset
        from mmcv import Config
        
        print("✅ Core training imports successful")
        
        # Test config loading
        config_path = "configs/s2anet_r50_fpn_1x_dota12_base.py"
        if Path(config_path).exists():
            try:
                cfg = Config.fromfile(config_path)
                print("✅ Config file loads successfully")
                return True
            except Exception as e:
                print(f"❌ Config loading failed: {e}")
                return False
        else:
            print("⚠️  Config file not found - cannot test config loading")
            return False
            
    except ImportError as e:
        print(f"❌ Training import failed: {e}")
        return False

def main():
    """Run all verification checks."""
    print("=" * 60)
    print("🚀 S2ANet Clean Environment Verification")
    print("=" * 60)
    
    checks = [
        ("Package Imports", check_imports),
        ("Config File", check_config_file), 
        ("Essential Directories", check_essential_directories),
        ("No Local Package Conflicts", check_no_local_packages),
        ("Training Readiness", check_training_readiness)
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ {check_name} check failed with error: {e}")
            results.append((check_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 VERIFICATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {check_name}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 Environment is clean and ready for S2ANet training!")
        print("\nTo start training:")
        print("conda activate s2anet")
        print("python tools/train.py configs/s2anet_r50_fpn_1x_dota12_base.py --work_dir work_dirs/s2anet_base12")
    else:
        print(f"\n⚠️  Please address the {total - passed} failed check(s) before training")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
