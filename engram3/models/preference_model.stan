data {
    int<lower=1> N;                     // number of comparisons
    int<lower=1> P;                     // number of participants
    int<lower=1> F;                     // number of features
    matrix[N, F] X1;                    // features for first bigram in each comparison
    matrix[N, F] X2;                    // features for second bigram in each comparison
    array[N] int<lower=1> participant;  // participant ID for each comparison
    array[N] int<lower=0, upper=1> y;   // observed preferences (1 if first bigram preferred)
    // Hyperparameters
    real<lower=0> feature_scale;        // prior scale for feature weights
    real<lower=0> participant_scale;    // prior scale for participant effects
}

parameters {
    vector[F] beta;                     // feature weights
    vector[P] participant_raw;          // participant effects (non-centered)
    real<lower=0> tau;                  // participant effect scale
}

transformed parameters {
    vector[P] participant_effect = participant_raw * tau;  // centered participant effects
    vector[N] comfort_diff;             // difference in comfort scores
    for (n in 1:N) {
        real comfort1 = dot_product(X1[n], beta) + participant_effect[participant[n]];
        real comfort2 = dot_product(X2[n], beta) + participant_effect[participant[n]];
        comfort_diff[n] = comfort1 - comfort2;
    }
}

model {
    // Priors
    beta ~ normal(0, feature_scale);            // prior on feature weights
    participant_raw ~ std_normal();             // non-centered participant effects
    tau ~ cauchy(0, participant_scale);         // prior on participant effect scale
    // Likelihood
    y ~ bernoulli_logit(comfort_diff);         // logistic model for preferences
}

generated quantities {
    // Predicted probabilities
    vector[N] p_pred = inv_logit(comfort_diff);
    // Log likelihood for model comparison
    vector[N] log_lik;
    for (n in 1:N) {
        log_lik[n] = bernoulli_logit_lpmf(y[n] | comfort_diff[n]);
    }
    // Comfort score function for new bigrams
    // (implemented in Python wrapper)
}