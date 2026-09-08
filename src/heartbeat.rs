use std::time::Duration;

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn chance(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn range_inclusive(&mut self, low: i64, high: i64) -> i64 {
        let span = (high - low) as u64 + 1;
        low + (self.next_u64() % span) as i64
    }
}

const ACT_CHANCE: f64 = 0.5;
const SLEEP_CHANCE: f64 = 0.01;
const SLEEP_MIN_MINUTES: i64 = 15;
const SLEEP_MAX_MINUTES: i64 = 120;

#[derive(Clone, Copy)]
pub struct SocialPace {
    pub idle_return_chance: f64,
    pub idle_return_min: Duration,
    pub idle_return_max: Duration,
    pub idle_return_for: Duration,
}

impl Default for SocialPace {
    fn default() -> Self {
        Self {
            idle_return_chance: 0.35,
            idle_return_min: Duration::from_secs(2 * 60),
            idle_return_max: Duration::from_secs(8 * 60),
            idle_return_for: Duration::from_secs(90),
        }
    }
}

#[derive(Clone, Copy)]
pub struct PresencePlan {
    pub idle_return_after: Option<Duration>,
    pub idle_return_for: Duration,
}

pub struct Heartbeat {
    random: SplitMix64,
    social_pace: SocialPace,
    // Unix seconds until which she stays asleep; 0 means awake.
    asleep_until: i64,
}

impl Heartbeat {
    pub fn new(seed: u64) -> Self {
        Self {
            random: SplitMix64::new(seed),
            social_pace: SocialPace::default(),
            asleep_until: 0,
        }
    }

    /// One heartbeat at unix time `now`. `true` means act this tick.
    pub fn tick(&mut self, now: i64) -> bool {
        if now < self.asleep_until {
            return false;
        }
        if self.random.chance() < SLEEP_CHANCE {
            let nap = self
                .random
                .range_inclusive(SLEEP_MIN_MINUTES, SLEEP_MAX_MINUTES);
            self.asleep_until = now + nap * 60;
            return false;
        }
        self.random.chance() < ACT_CHANCE
    }

    /// A message just arrived; end any nap so she can decide to respond to it.
    pub fn wake(&mut self) {
        self.asleep_until = 0;
    }

    /// Decide whether her next quiet stretch gets a brief, unprompted return.
    pub fn presence_plan(&mut self) -> PresencePlan {
        let pace = self.social_pace;
        let idle_return_after = (self.random.chance() < pace.idle_return_chance.clamp(0.0, 1.0))
            .then(|| {
                let min = pace.idle_return_min.as_secs() as i64;
                let max = pace.idle_return_max.max(pace.idle_return_min).as_secs() as i64;
                Duration::from_secs(self.random.range_inclusive(min, max) as u64)
            });
        PresencePlan {
            idle_return_after,
            idle_return_for: pace.idle_return_for,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The same act-rate band the old selfcheck asserted: a wrong PRNG or a
    // miscounted draw drifts off it. Two draws per waking tick, so the rate is
    // ~0.99 * 0.5.
    #[test]
    fn act_rate_lands_in_band() {
        let mut heartbeat = Heartbeat::new(42);
        let mut acts = 0;
        let n = 20_000;
        let mut now = 0;
        for _ in 0..n {
            now += 100_000; // jump past any nap so ticks stay independent
            if heartbeat.tick(now) {
                acts += 1;
            }
        }
        let rate = acts as f64 / n as f64;
        assert!(rate > 0.45 && rate < 0.54, "act-rate {rate} out of band");
    }

    #[test]
    fn a_nap_suppresses_acting_until_woken() {
        let mut heartbeat = Heartbeat::new(1);
        heartbeat.asleep_until = 10_000;
        assert!(!heartbeat.tick(0), "asleep, so never acts"); // now < asleep_until
        heartbeat.wake();
        assert_eq!(heartbeat.asleep_until, 0, "wake clears the nap");
    }
}
