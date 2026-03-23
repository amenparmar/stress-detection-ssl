#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# prefer ./venv, then .venv, then system python
if [ -x "$SCRIPT_DIR/venv/bin/python" ]; then
  VENV_PYTHON="$SCRIPT_DIR/venv/bin/python"
elif [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
  VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"
else
  VENV_PYTHON="$(command -v python3 || command -v python)"
fi
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

clear

# default batch size suggestion
DEFAULT_BATCH=200

echo "SELECT MODE:"
echo "1. Test Run (Mock Data)"
echo "2. Pre-train (Real WESAD Data - 500 epochs)"
echo "3. Calculate Model Accuracy (Evaluate - Standard Encoder)"
echo "4. Train Ensemble (5 models for better accuracy)"
echo "5. Train Multi-Modal Fusion (Separate encoders per modality)"
echo "6. FULL PIPELINE - All Improvements Combined"
echo "7. SMOTE Oversampling (Fix Class Imbalance)"
echo "8. Leave-One-Subject-Out CV (Gold Standard Evaluation)"
echo "9. Domain Adversarial Training (DANN - Subject-Invariant)"
echo "10. Latent Trajectory Analysis (Continuous Monitoring)"
echo "11. Subject-Invariant Loss Training (MMD + CORAL + Contrastive)"
echo "12. COMBINED ADVANCED - MAXIMUM PERFORMANCE (DANN + Multi-Modal)"
echo "13. 🏆 ULTIMATE PERFORMANCE - ALL TECHNIQUES + ENSEMBLE (85-88% Expected)"
echo "14. 📊 BENCHMARK ALL MODELS - Run and rank all configurations"
echo "15. 🚀 ADVANCED BENCHMARK - Test SMOTE, DANN, Invariant, Ultimate"
echo ""
echo "99. RESET CACHE & CHECKPOINTS (Clear pycache/models)"
echo ""
read -p "Enter choice (1-15 or 99): " choice

# Prompt for batch size for modes that run training/evaluation (1-15)
if [[ "$choice" =~ ^([1-9]|1[0-5])$ ]]; then
  read -p "Enter batch size [${DEFAULT_BATCH}]: " SELECTED_BATCH
  SELECTED_BATCH=${SELECTED_BATCH:-$DEFAULT_BATCH}
fi

run_cmd() {
  echo "Running: $1"
  cd "$PARENT_DIR"
  "$VENV_PYTHON" -m stress_detection.main $1
}

case "$choice" in
  1) run_cmd "--mode test_run --batch_size ${SELECTED_BATCH}" ;;
  2) run_cmd "--mode pretrain --epochs 500 --batch_size ${SELECTED_BATCH}" ;;
  3) run_cmd "--mode evaluate --epochs 100 --batch_size ${SELECTED_BATCH}" ;;
  4) run_cmd "--mode ensemble --epochs 100 --batch_size ${SELECTED_BATCH}" ;;
  5) run_cmd "--mode multimodal --epochs 100 --batch_size ${SELECTED_BATCH}" ;;
  6)
    echo ""
    echo "========================================"
    echo "  FULL PIPELINE - Maximum Accuracy"
    echo "========================================"
    echo "This will run:"
    echo "1. Pre-training with 500 epochs"
    echo "2. Multi-Modal Ensemble with 5 fusion models"
    echo "Expected accuracy: 85-88 percent"
    echo "Estimated time: 3-4 hours on CPU or 20-30 minutes on GPU"
    echo ""
    read -p "Press Enter to continue..."
    run_cmd "--mode pretrain --epochs 500 --batch_size ${SELECTED_BATCH}"
    run_cmd "--mode multimodal_ensemble --epochs 100 --batch_size ${SELECTED_BATCH}"
    echo "========================================"
    echo "  FULL PIPELINE COMPLETE"
    echo "========================================"
    ;;
  7) run_cmd "--mode smote --epochs 100 --batch_size ${SELECTED_BATCH}" ;;
  8)
    echo ""
    echo "========================================"
    echo "  LEAVE-ONE-SUBJECT-OUT CV"
    echo "========================================"
    echo "This will train and test on EACH subject"
    echo "Estimated time: 3-6 hours (15 subjects)"
    echo ""
    read -p "Press Enter to continue..."
    run_cmd "--mode loso --epochs 100 --batch_size ${SELECTED_BATCH}"
    ;;
  9)
    echo ""
    echo "========================================"
    echo "  DOMAIN ADVERSARIAL TRAINING (DANN)"
    echo "========================================"
    echo "Subject-invariant feature learning"
    echo "Expected improvement: 74% -> 78-82% LOSO accuracy"
    echo ""
    read -p "Press Enter to continue..."
    run_cmd "--mode dann --epochs 100 --batch_size ${SELECTED_BATCH}"
    ;;
  10)
    echo ""
    echo "========================================"
    echo "  LATENT TRAJECTORY ANALYSIS"
    echo "========================================"
    echo "Continuous stress monitoring"
    echo "Personalized baselines per subject"
    echo ""
    read -p "Press Enter to continue..."
    run_cmd "--mode trajectory --epochs 100 --batch_size ${SELECTED_BATCH}"
    ;;
  11)
    echo ""
    echo "========================================"
    echo "  SUBJECT-INVARIANT LOSS TRAINING"
    echo "========================================"
    echo "Using MMD + CORAL + Contrastive losses"
    echo "Expected improvement: 3-7% accuracy gain"
    echo ""
    read -p "Press Enter to continue..."
    run_cmd "--mode invariant --epochs 100 --batch_size ${SELECTED_BATCH}"
    ;;
  12)
    echo ""
    echo "========================================"
    echo "  COMBINED ADVANCED - MAXIMUM PERFORMANCE"
    echo "========================================"
    echo "Combines:"
    echo "- Domain Adversarial Training"
    echo "- Multi-Modal Fusion"
    echo "- Subject-Invariant Losses"
    echo "Expected: 82-86% LOSO accuracy"
    echo "Estimated time: 2-3 hours on GPU"
    echo ""
    read -p "Press Enter to continue..."
    run_cmd "--mode combined --epochs 100 --batch_size ${SELECTED_BATCH}"
    ;;
  13)
    echo ""
    echo "========================================"
    echo "  🏆 ULTIMATE PERFORMANCE PIPELINE 🏆"
    echo "========================================"
    echo ""
    echo "This is the MOST POWERFUL configuration:"
    echo ""
    echo "Stage 1: SSL Pre-training (500 epochs)"
    echo "Stage 2: Ensemble of 5 Ultimate Models"
    echo "         - Multi-Modal Fusion"
    echo "         - Domain Adversarial (DANN)"
    echo "         - Subject-Invariant Losses"
    echo "         - Trajectory Analysis"
    echo "         - Temporal Consistency"
    echo "Stage 3: Ensemble Evaluation"
    echo ""
    echo "Expected: 85-88% LOSO accuracy"
    echo "Current Baseline: 74.35%"
    echo "Improvement: +11-14% absolute gain!"
    echo ""
    echo "Estimated time: 6-8 hours on RTX 5070 Ti GPU"
    echo ""
    read -p "Press Enter to continue..."
    run_cmd "--mode ultimate --epochs 100 --batch_size ${SELECTED_BATCH}"
    ;;
  14)
    echo ""
    echo "========================================"
    echo "  📊 BENCHMARK ALL MODELS"
    echo "========================================"
    echo ""
    echo "This will run and rank ALL configurations:"
    echo "  1. Baseline (SSL + Classifier)"
    echo "  2. Multi-Modal Fusion"
    echo "  3. Multi-Modal Ensemble (5 models)"
    echo "  4. SMOTE Oversampling"
    echo "  5. DANN (Domain Adversarial)"
    echo "  6. Trajectory Analysis"
    echo "  7. Subject-Invariant Losses"
    echo "  8. Combined (DANN + Multi-Modal)"
    echo "  9. Ultimate (All Techniques)"
    echo ""
    echo "Estimated Time: 15-20 hours (full) or 3-4 hours (quick mode)"
    echo "========================================"
    echo ""
    read -p "Press Enter to continue..."
    run_cmd "--mode benchmark --batch_size ${SELECTED_BATCH}"
    ;;
  15)
    echo ""
    echo "========================================"
    echo "  🚀 ADVANCED BENCHMARK"
    echo "========================================"
    echo ""
    echo "This will run and rank advanced techniques:"
    echo "  1. SMOTE Oversampling"
    echo "  2. DANN (Domain Adversarial)"
    echo "  3. Subject-Invariant Loss"
    echo "  4. Ultimate Performance"
    echo ""
    echo "Estimated Time: 20-25 hours (full) or 4-6 hours (quick mode)"
    echo "========================================"
    echo ""
    read -p "Press Enter to continue..."
    run_cmd "--mode advanced_benchmark --batch_size ${SELECTED_BATCH}"
    ;;
  99)
    echo ""
    echo "========================================"
    echo "  RESET CACHE & CHECKPOINTS"
    echo "========================================"
    echo ""
    echo "1. Clear Python Cache (__pycache__) - Safe, recommended"
    echo "2. Clear Saved Models (checkpoints) - DESTRUCTIVE (You lose training!)"
    echo "3. Clear BOTH"
    echo "4. Cancel"
    echo ""
    read -p "Select (1-4): " clean_choice

    case "$clean_choice" in
      1)
        echo "Cleaning __pycache__..."
        find "$SCRIPT_DIR/stress_detection" -type d -name "__pycache__" -exec rm -rf {} +
        echo "Done."
        ;;
      2)
        read -p "WARNING: Delete all models? (y/N): " confirm
        if [[ "$confirm" == "y" || "$confirm" == "Y" ]]; then
          rm -rf "$SCRIPT_DIR/stress_detection/models"/*
          echo "Models deleted."
        else
          echo "Cancelled."
        fi
        ;;
      3)
        echo "Cleaning __pycache__..."
        find "$SCRIPT_DIR/stress_detection" -type d -name "__pycache__" -exec rm -rf {} +
        read -p "WARNING: Delete all models? (y/N): " confirm
        if [[ "$confirm" == "y" || "$confirm" == "Y" ]]; then
          rm -rf "$SCRIPT_DIR/stress_detection/models"/*
          echo "Models deleted."
        else
          echo "Cancelled."
        fi
        ;;
      *)
        echo "Cancelled."
        ;;
    esac
    ;;
  *)
    echo "Invalid choice. Please run again and select 1-15 or 99."
    ;;
esac

echo ""
read -p "Press Enter to exit..."
