#pragma once

template <typename Sessions, typename Owner, typename ReadState>
static bool xdna_rec_prepare_host(Sessions & sessions, Owner owner, ReadState read_state) {
    for (auto & entry : sessions) {
        auto * session = entry.second;
        if (!session || !session->seeded || session->owner != owner) {
            continue;
        }
        if (!read_state(*session)) {
            return false;
        }
        session->seeded = false;
    }
    return true;
}
