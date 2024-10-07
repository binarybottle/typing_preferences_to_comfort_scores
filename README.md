# Bigram Typing Study 
-- jspsych website via Prolific and python analysis --

Author: Arno Klein (binarybottle.com)

GitHub repository: binarybottle/bigram-typing-comfort-experiment

License: Apache v2.0 

The purpose of the scripts in this repository is to determine how comfortable 
different pairs of keys are to type on computer keyboards, 
ultimately to inform the design of future keyboard layouts.

  - experiment.js: script to present and collect bigram typing data via a website.
  
    - Step 1. Present consent and instructions.
    - Step 2. Present pair of bigrams to be typed repeatedly between random text.
    - Step 3. Present slider bar to indicate which bigram is easier to type.
    - Step 4. Collect timing and bigram preference data.
    - Step 5. Store data in OSF.io data repository.
    - Step 6. Send participant back to the Prolific crowdsourcing platform.

  - analyze_bigram_prolific_study_data.py: script to analyze bigram typing data.
    - Input: csv tables of summary data, easy choice (improbable) bigram pairs, and remove pairs.
    - Output: csv tables and plots.
    - Step 1. Load and combine the data collected from the study above.
    - Step 2. Filter users by improbable or inconsistent choice thresholds.
    - Step 3. Analyze bigram typing times. 
    - Step 4. Score choices by slider values.
    - Step 5. Choose winning bigrams for bigram pairs.
