// Hierarchical Bayesian model for keyboard layout preferences
// Includes main features and control features (e.g., bigram frequency)
data {
    int<lower=1> N;                     // Number of preferences
    int<lower=1> P;                     // Number of participants
    int<lower=1> F;                     // Number of main features
    int<lower=0> C;                     // Number of control features
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
    vector[F] beta;                     // Main feature weights
    vector[C] gamma;                    // Control feature weights (nuisance parameters)
    vector[P] z;                        // Participant random effects (standardized)
    real<lower=0> tau;                  // Random effects scale
}

transformed parameters {
    vector[P] participant_effects;       // Scaled participant effects
    vector[N] mu;                       // Linear predictor
    
    // Scale participant effects
    participant_effects = tau * z;
    
    // Compute linear predictor
    for (n in 1:N) {
        // Main features contribution
        mu[n] = dot_product(X1[n], beta) - dot_product(X2[n], beta);
        
        // Add control features contribution (if any)
        if (C > 0) {
            mu[n] = mu[n] + dot_product(C1[n], gamma) - dot_product(C2[n], gamma);
        }
        
        // Add participant effect
        mu[n] = mu[n] + participant_effects[participant[n]];
    }
}

model {
    // Priors for main feature weights
    beta ~ normal(0, feature_scale);
    
    // Tighter priors for control features (nuisance parameters)
    if (C > 0) {
        gamma ~ normal(0, feature_scale * 0.5);  // Tighter prior for control features
    }
    
    // Priors for participant effects
    z ~ std_normal();                   // Standardized random effects
    tau ~ exponential(1/participant_scale);  // Scale of random effects
    
    // Likelihood
    y ~ bernoulli_logit(mu);
}

generated quantities {
    vector[N] log_lik;                  // Log likelihood for each observation
    vector[N] y_pred;                   // Predicted probabilities
    
    for (n in 1:N) {
        log_lik[n] = bernoulli_logit_lpmf(y[n] | mu[n]);
        y_pred[n] = inv_logit(mu[n]);
    }
}