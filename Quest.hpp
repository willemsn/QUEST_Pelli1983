#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <numeric>
#include <limits>

/**
 * Parameters for the Weibull psychometric function and trial grid setup.
 */
struct QuestParams {
    double t_prior_mean = 0.0;    // Prior threshold mean (dB)
    double t_prior_sd   = 12.0;   // Prior threshold standard deviation (dB)
    double grain        = 1.0;    // Step size along log intensity / dB axis (dB)
    double range_db     = 40.0;   // Total evaluation range centered on t_prior_mean (dB)
    
    // Weibull Psychometric Function parameters:
    double beta    = 3.5;   // Slope parameter
    double gamma   = 0.5;   // Chance success rate (e.g. 0.5 for 2AFC, 0.0 for Yes/No)
    double delta   = 0.01;  // Lapse rate / error asymptote at high intensity
    double epsilon = 1.5;   // Sweat factor offset (1.5 dB for 2AFC, 1.15 dB for Yes/No)
};

/**
 * Structure holding confidence bounds computed via likelihood-ratio testing.
 */
struct QuestConfidenceInterval {
    double lower_bound;
    double upper_bound;
    double confidence_level;
};

class Quest {
public:
    explicit Quest(const QuestParams& params = QuestParams())
        : params_(params) 
    {
        if (params_.grain <= 0.0) {
            throw std::invalid_argument("Grain step size must be positive.");
        }
        if (params_.t_prior_sd <= 0.0) {
            throw std::invalid_argument("Prior SD must be positive.");
        }
        initialize_grids();
    }

    /**
     * Resets the QUEST state back to initial log prior Q0.
     */
    void reset() {
        q_ = q0_;
        trial_count_ = 0;
    }

    /**
     * Recommends the next stimulus intensity x_next (dB) at current posterior mode T_post.
     */
    double get_recommended_intensity() const {
        auto max_it = std::max_element(q_.begin(), q_.end());
        size_t idx = std::distance(q_.begin(), max_it);
        return t_grid_[idx];
    }

    /**
     * Updates log posterior density Q(T) given stimulus intensity and trial outcome.
     * @param intensity Stimulus test intensity x (dB).
     * @param success True (1) if response was correct/yes, False (0) if incorrect/no.
     */
    void update(double intensity, bool success) {
        for (size_t i = 0; i < grid_size_; ++i) {
            double delta_t = t_grid_[i] - intensity; // u = T - x
            
            // Map delta_t to table index
            int idx = static_cast<int>(std::round(delta_t / params_.grain)) + rel_steps_;
            idx = std::clamp(idx, 0, static_cast<int>(rel_grid_size_ - 1));
            
            q_[i] += success ? s_table_[idx] : f_table_[idx];
        }
        trial_count_++;
    }

    /**
     * Returns Maximum A Posteriori (MAP) threshold estimate (T_post).
     */
    double get_map_estimate() const {
        return get_recommended_intensity();
    }

    /**
     * Returns Maximum Likelihood Estimate (MLE) threshold (T_like), 
     * isolating likelihood L(T) = Q(T) - Q0(T) per Watson & Pelli rules.
     */
    double get_mle_estimate() const {
        std::vector<double> l = get_log_likelihood();
        auto max_it = std::max_element(l.begin(), l.end());
        size_t idx = std::distance(l.begin(), max_it);
        return t_grid_[idx];
    }

    /**
     * Returns expected mean of posterior PDF.
     */
    double get_posterior_mean() const {
        double max_q = *std::max_element(q_.begin(), q_.end());
        double sum_weights = 0.0;
        double weighted_t_sum = 0.0;

        for (size_t i = 0; i < grid_size_; ++i) {
            double w = std::exp(q_[i] - max_q);
            sum_weights += w;
            weighted_t_sum += w * t_grid_[i];
        }

        return weighted_t_sum / sum_weights;
    }

    /**
     * Computes confidence interval for threshold estimate using asymptotic chi-square 
     * distribution of -2 * log likelihood ratio (-2 * [L(T_like) - L(T)] <= chi^2_alpha).
     */
    QuestConfidenceInterval get_confidence_interval(double confidence_level = 0.95) const {
        std::vector<double> l = get_log_likelihood();
        auto max_it = std::max_element(l.begin(), l.end());
        size_t mle_idx = std::distance(l.begin(), max_it);
        double l_max = *max_it;

        // 1 df critical value threshold: 0.5 * chi^2(1, alpha)
        double crit_val = 1.920725; // default 95% (chi^2 = 3.84145)
        if (std::abs(confidence_level - 0.975) < 0.005) {
            crit_val = 2.51189; // 97.5% (chi^2 = 5.02389)
        } else if (std::abs(confidence_level - 0.90) < 0.005) {
            crit_val = 1.35277; // 90% (chi^2 = 2.70554)
        }

        size_t lower_idx = 0;
        for (size_t i = mle_idx; i > 0; --i) {
            if ((l_max - l[i]) > crit_val) {
                lower_idx = i;
                break;
            }
        }

        size_t upper_idx = grid_size_ - 1;
        for (size_t i = mle_idx; i < grid_size_; ++i) {
            if ((l_max - l[i]) > crit_val) {
                upper_idx = i;
                break;
            }
        }

        return {t_grid_[lower_idx], t_grid_[upper_idx], confidence_level};
    }

    /**
     * Checks if maximum likelihood peak sits at grid boundary (unbounded condition).
     */
    bool is_unbounded() const {
        std::vector<double> l = get_log_likelihood();
        auto max_it = std::max_element(l.begin(), l.end());
        size_t idx = std::distance(l.begin(), max_it);
        return (idx == 0 || idx == grid_size_ - 1);
    }

    size_t get_trial_count() const { return trial_count_; }
    const std::vector<double>& get_grid() const { return t_grid_; }
    const std::vector<double>& get_log_posterior() const { return q_; }
    
    std::vector<double> get_log_likelihood() const {
        std::vector<double> l(grid_size_);
        for (size_t i = 0; i < grid_size_; ++i) {
            l[i] = q_[i] - q0_[i];
        }
        return l;
    }

private:
    void initialize_grids() {
        int half_steps = static_cast<int>(std::round((params_.range_db / 2.0) / params_.grain));
        grid_size_ = 2 * half_steps + 1;
        
        t_grid_.resize(grid_size_);
        q0_.resize(grid_size_);

        for (int i = 0; i < static_cast<int>(grid_size_); ++i) {
            double t = params_.t_prior_mean + (i - half_steps) * params_.grain;
            t_grid_[i] = t;
            
            // Log Gaussian prior density Q0(T) = -0.5 * ((T - T_prior) / SD)^2
            double norm_diff = (t - params_.t_prior_mean) / params_.t_prior_sd;
            q0_[i] = -0.5 * norm_diff * norm_diff;
        }
        
        q_ = q0_;
        trial_count_ = 0;

        // Relative index offset grid: u = T - x ranges from -2*half_steps to +2*half_steps
        rel_steps_ = 2 * half_steps;
        rel_grid_size_ = 2 * rel_steps_ + 1;
        
        s_table_.resize(rel_grid_size_);
        f_table_.resize(rel_grid_size_);

        for (int i = 0; i < static_cast<int>(rel_grid_size_); ++i) {
            double delta_t = (i - rel_steps_) * params_.grain; // u = T - x
            double rel_x = -delta_t;                          // x - T

            // Weibull Psi(rel_x) = 1 - (1 - gamma) * exp(-10^( (beta/20) * (rel_x + epsilon) ))
            double exponent = std::pow(10.0, (params_.beta / 20.0) * (rel_x + params_.epsilon));
            double psi = 1.0 - (1.0 - params_.gamma) * std::exp(-exponent);
            
            // Apply upper asymptote (1 - delta)
            if (psi > 1.0 - params_.delta) {
                psi = 1.0 - params_.delta;
            }
            if (psi < 1e-12) {
                psi = 1e-12;
            }

            // S(u) = ln Psi(-u), F(u) = ln(1 - Psi(-u))
            s_table_[i] = std::log(psi);
            f_table_[i] = std::log(1.0 - psi);
        }
    }

    QuestParams params_;
    size_t grid_size_ = 0;
    int rel_steps_ = 0;
    size_t rel_grid_size_ = 0;
    size_t trial_count_ = 0;

    std::vector<double> t_grid_;
    std::vector<double> q0_;
    std::vector<double> q_;
    std::vector<double> s_table_;
    std::vector<double> f_table_;
};
