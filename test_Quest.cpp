#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "Quest.hpp"
#include <random>

using Catch::Matchers::WithinAbs;

TEST_CASE("QUEST Initialization and Canonical Psychometric Values", "[quest]") {
    QuestParams params;
    params.t_prior_mean = 0.0;
    params.t_prior_sd   = 12.0;
    params.grain        = 1.0;
    params.range_db     = 40.0;
    params.beta         = 3.5;
    params.gamma        = 0.5;
    params.delta        = 0.01;
    params.epsilon      = 1.5;

    Quest quest(params);

    SECTION("Initial trial placement starts at prior mode") {
        REQUIRE_THAT(quest.get_recommended_intensity(), WithinAbs(0.0, 1e-6));
        REQUIRE_THAT(quest.get_map_estimate(), WithinAbs(0.0, 1e-6));
        REQUIRE(quest.get_trial_count() == 0);
    }

    SECTION("Log Prior Q0 symmetric around prior mean") {
        const auto& q = quest.get_log_posterior();
        REQUIRE(q.size() == 41); // -20 to +20 inclusive
        
        // Mode index at center (20)
        REQUIRE_THAT(q[20], WithinAbs(0.0, 1e-6));
        
        // Check symmetry around prior mean
        REQUIRE_THAT(q[10], WithinAbs(q[30], 1e-6));
    }
}

TEST_CASE("QUEST Trial Updating and Log Posterior Shifts", "[quest]") {
    QuestParams params;
    params.t_prior_mean = -6.0;
    params.t_prior_sd   = 8.0;
    params.grain        = 1.0;
    params.range_db     = 40.0;

    Quest quest(params);

    SECTION("Successful trial proves higher sensitivity and lowers threshold estimate") {
        double initial_rec = quest.get_recommended_intensity();
        quest.update(initial_rec, true); // Correct response
        
        REQUIRE(quest.get_trial_count() == 1);
        // Correct response at -6 dB means observer threshold is lower (more sensitive)
        REQUIRE(quest.get_recommended_intensity() <= initial_rec);
    }

    SECTION("Failed trial proves lower sensitivity and raises threshold estimate") {
        double initial_rec = quest.get_recommended_intensity();
        quest.update(initial_rec, false); // Incorrect response
        
        REQUIRE(quest.get_trial_count() == 1);
        // Incorrect response at -6 dB means observer threshold is higher (less sensitive)
        REQUIRE(quest.get_recommended_intensity() >= initial_rec);
    }
}

TEST_CASE("QUEST Synthetic Observer Simulation and Threshold Convergence", "[quest]") {
    QuestParams params;
    params.t_prior_mean = 0.0;
    params.t_prior_sd   = 8.0;
    params.grain        = 1.0;
    params.range_db     = 40.0;
    params.beta         = 3.5;
    params.gamma        = 0.5;
    params.delta        = 0.01;
    params.epsilon      = 1.5;

    Quest quest(params);

    const double true_threshold = 4.0; // Ground truth threshold in dB
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    // Simulate 40 trials
    for (int trial = 0; trial < 40; ++trial) {
        double test_intensity = quest.get_recommended_intensity();
        double rel_x = test_intensity - true_threshold;
        
        // Weibull success probability for simulated observer
        double p_success = 1.0 - (1.0 - params.gamma) * 
            std::exp(-std::pow(10.0, (params.beta / 20.0) * (rel_x + params.epsilon)));
        p_success = std::min(1.0 - params.delta, p_success);

        bool response = (dist(rng) < p_success);
        quest.update(test_intensity, response);
    }

    SECTION("Final MLE and MAP estimates converge within 2.0 dB of ground truth") {
        double mle = quest.get_mle_estimate();
        double map = quest.get_map_estimate();
        double mean = quest.get_posterior_mean();

        REQUIRE_THAT(mle, WithinAbs(true_threshold, 2.0));
        REQUIRE_THAT(map, WithinAbs(true_threshold, 2.0));
        REQUIRE_THAT(mean, WithinAbs(true_threshold, 2.0));
    }

    SECTION("Confidence interval encompasses ground truth threshold") {
        auto ci = quest.get_confidence_interval(0.95);
        REQUIRE(ci.lower_bound <= true_threshold);
        REQUIRE(ci.upper_bound >= true_threshold);
    }
}

TEST_CASE("QUEST Reset functionality", "[quest]") {
    Quest quest;
    quest.update(0.0, true);
    quest.update(1.0, false);
    
    REQUIRE(quest.get_trial_count() == 2);
    
    quest.reset();
    
    REQUIRE(quest.get_trial_count() == 0);
    REQUIRE_THAT(quest.get_recommended_intensity(), WithinAbs(0.0, 1e-6));
}
