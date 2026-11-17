#include "../src/ggml-xdna/xdna-rec-host.h"

#include <cstdio>
#include <map>

struct recurrent_session {
    bool seeded;
    int owner;
    float host_state;
    float device_state;
};

static float step(float state, float input) {
    return state * 0.5f + input;
}

static bool check_transition() {
    recurrent_session active = { true, 1, 0.0f, step(0.0f, 2.0f) };
    recurrent_session other = { true, 2, 0.0f, 9.0f };
    std::map<int, recurrent_session *> sessions = { {0, &active}, {1, &other}, {2, nullptr} };
    int reads = 0;
    const auto read = [&reads](recurrent_session & s) {
        ++reads;
        s.host_state = s.device_state;
        return true;
    };
    if (!xdna_rec_prepare_host(sessions, 1, read) || active.seeded || !other.seeded || reads != 1) {
        return false;
    }
    active.host_state = step(active.host_state, 3.0f);
    active.host_state = step(active.host_state, 4.0f);
    if (!active.seeded) {
        active.device_state = active.host_state;
        active.seeded = true;
    }
    active.device_state = step(active.device_state, 5.0f);
    const float reference = step(step(step(step(0.0f, 2.0f), 3.0f), 4.0f), 5.0f);
    return active.device_state == reference && other.host_state == 0.0f && other.device_state == 9.0f;
}

static bool check_failed_readback() {
    recurrent_session active = { true, 1, 0.0f, 2.0f };
    std::map<int, recurrent_session *> sessions = { {0, &active} };
    const bool ready = xdna_rec_prepare_host(sessions, 1, [](recurrent_session &) { return false; });
    return !ready && active.seeded && active.host_state == 0.0f;
}

static bool check_unseeded_session() {
    recurrent_session active = { false, 1, 3.0f, 2.0f };
    std::map<int, recurrent_session *> sessions = { {0, &active} };
    const bool ready = xdna_rec_prepare_host(sessions, 1, [](recurrent_session &) { return false; });
    return ready && !active.seeded && active.host_state == 3.0f;
}

int main() {
    if (!check_transition() || !check_failed_readback() || !check_unseeded_session()) {
        std::fprintf(stderr, "test-xdna-rec-host: failed\n");
        return 1;
    }
    std::puts("test-xdna-rec-host: passed");
    return 0;
}
