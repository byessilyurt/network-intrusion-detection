#!/bin/bash

echo "🧹 Cleaning up unnecessary files..."

# Remove debug and interim reports
git rm -f AUTOENCODER_DEBUG_SUMMARY.md
git rm -f DEBUGGING_REPORT.md
git rm -f FULL_DATASET_FINAL_REPORT.md
git rm -f MULTI_ATTACK_EXECUTIVE_SUMMARY.md
git rm -f MULTI_ATTACK_TEST_REPORT.md
git rm -f OCSVM_TUNING_FINAL_REPORT.md
git rm -f OCSVM_TUNING_INTERIM_REPORT.md

# Remove backup and redundant files
git rm -f CLAUDE.md.backup_before_correction
git rm -f README_RESULTS.md
git rm -f RESULTS_SUMMARY.txt
git rm -f PROJECT_PLAN.md
git rm -f DEPLOYMENT.md

# Remove data documentation (data not in repo anyway)
git rm -f data/DOWNLOAD_INSTRUCTIONS.md
git rm -f data/README.md

# Remove one-off scripts
git rm -f create_debug_visualization.py
git rm -f debug_autoencoder.py
git rm -f implement_shap.py
git rm -f implement_shap_ocsvm.py
git rm -f retune_ocsvm_200k.py
git rm -f save_production_artifacts.py

# Remove all result files except .gitkeep
git rm -f results/*.txt
git rm -f results/*.pkl
git rm -f results/*.json

echo "✅ Cleanup complete!"
echo ""
echo "Files removed:"
git status --short
