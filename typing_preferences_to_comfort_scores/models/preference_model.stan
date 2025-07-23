// Hierarchical Bayesian model for keyboard layout preferences
// Allows for control-features-only case (no main features)
data {
    int<lower=1> N;                     // Number of preferences
    int<lower=1> P;                     // Number of participants
    int<lower=0> F;                     // Number of main features (can be 0)
    int<lower=1> C;                     // Number of control features (must be >= 1)
    int<lower=0,upper=1> has_main_features;  // Flag for whether main features exist
    matrix[N, F] X1;                    // Main features for first bigram
    matrix[N, F] X2;                    // Main features for second bigram
    matrix[N, C] C1;                    // Control features for first bigram
    matrix[N, C] C2;                    // Control features for second bigram
    array[N] int<lower=1,upper=P> participant;  // Participant IDs
    array[N] int<lower=0,upper=1> y;    // Preferences (1 if first bigram preferred)
    real<lower=0> feature_scale;        // Prior scale for feature weights
    real<lower=0> participant_scale;     // Prior scale for participant effects
}

parameters {
    vector[F] beta;                     // Main feature weights (empty if F=0)
    vector[C] gamma;                    // Control feature weights
    vector[P] z;                        // Participant random effects (standardized)
    real<lower=0> tau;                  // Random effects scale
}

transformed parameters {
    vector[P] participant_effects = tau * z;       // Scaled participant effects
    vector[N] mu;                       // Linear predictor
    
    // Compute linear predictor
    mu = rep_vector(0, N);  // Initialize
    
    // Add main feature contribution if they exist
    if (F > 0) {
        mu += X1 * beta - X2 * beta;
    }
    
    // Add control feature contribution (always present)
    mu += C1 * gamma - C2 * gamma;
    
    // Add participant effects
    for (n in 1:N) {
        mu[n] += participant_effects[participant[n]];
    }
}

model {
    // Always apply priors regardless of dimensions
    beta ~ normal(0, feature_scale);    // Safe even when F=0
    gamma ~ normal(0, feature_scale);    // Always has elements since C>0
    z ~ std_normal();                    // Always has P elements
    tau ~ exponential(1/participant_scale);
    
    // Likelihood
    y ~ bernoulli_logit(mu);
}

generated quantities {
    vector[N] log_lik;
    vector[N] y_pred;
    
    for (n in 1:N) {
        log_lik[n] = bernoulli_logit_lpmf(y[n] | mu[n]);
        y_pred[n] = inv_logit(mu[n]);  // Return probabilities
    }
}