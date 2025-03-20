for feature in typing_time same_finger sum_finger_values outward_roll rows_apart angle_apart adj_finger_diff_row sum_engram_position_values sum_row_position_values; do
    echo "Analyzing feature: $feature"
    poetry run python3 analyze_single_feature.py config.yaml $feature
    sleep 5  # Give system time to recover
done