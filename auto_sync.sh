#!/bin/bash

# Auto-sync script for GitHub
# This script commits and pushes all changes to GitHub

echo "========================================"
echo "Auto-syncing to GitHub..."
echo "========================================"
echo

# Check if we're inside a git repository
if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    echo "========================================"
    echo "ERROR: Not inside a Git repository"
    echo "========================================"
    exit 1
fi

# Check if there are any changes
if [[ -n $(git status --porcelain) ]]; then
    echo
    echo "Adding all changes..."
    git add .

    echo
    echo "Committing changes..."
    git commit -m "Auto-sync: $(date '+%Y-%m-%d %H:%M:%S')"

    if [ $? -eq 0 ]; then
        echo
        echo "Pushing to GitHub..."
        git push

        if [ $? -eq 0 ]; then
            echo
            echo "========================================"
            echo "Successfully synced to GitHub!"
            echo "========================================"
        else
            echo
            echo "========================================"
            echo "ERROR: Failed to push to GitHub"
            echo "Please check your internet connection and credentials"
            echo "========================================"
            exit 1
        fi
    else
        echo
        echo "No changes to commit"
    fi
else
    echo
    echo "No changes detected."
fi

echo
read -p "Press Enter to continue..."
