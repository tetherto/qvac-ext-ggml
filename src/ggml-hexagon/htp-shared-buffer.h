#pragma once

// Private RPC buffer implementation. Include after the session and RPC API
// declarations; the portable regression supplies those transport boundaries.
struct ggml_hexagon_shared_buffer {
    ggml_hexagon_session * sess;
    uint8_t *              base;
    size_t                 size;
    int                    fd;
    bool                   mapped;
    bool                   pinned;

    void mmap() {
        fastrpc_map_flags flags = this->pinned ? FASTRPC_MAP_FD : FASTRPC_MAP_FD_DELAYED;

        int err = fastrpc_mmap(sess->domain_id, this->fd, (void *) this->base, 0, this->size, flags);
        if (err != 0) {
            GGML_LOG_ERROR("ggml-hex: %s buffer mapping failed : domain_id %d size %zu fd %d error 0x%08x\n", sess->c_name(),
                    sess->domain_id, this->size, this->fd, (unsigned) err);
            throw std::runtime_error("ggml-hex: fastrpc_mmap failed (see log for details)");
        }

        HEX_VERBOSE("ggml-hex: %s mapped buffer: base %p size %zu fd %d pinned %u\n",
                sess->c_name(), (void *) this->base, this->size, this->fd, pinned);

        this->mapped = true;
    }

    void unmap() {
        if (!this->mapped) return;

        if (!this->pinned) {
            // HTP might still hold a reference, tell it drop it
            htp_iface_munmap(sess->handle, this->fd);
        }

        fastrpc_munmap(sess->domain_id, this->fd, (void *) this->base, this->size);

        HEX_VERBOSE("ggml-hex: %s unmapped buffer: base %p size %zu fd %d\n", sess->c_name(),
                (void *) this->base, size, this->fd);

        this->mapped = false;
        this->fd     = -1;
    }

    void alloc(size_t size) {
        if (this->base) return;

        this->base = (uint8_t *) rpcmem_alloc2(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, size);
        if (!this->base) {
            GGML_LOG_ERROR("ggml-hex: %s failed to allocate buffer : size %zu\n", sess->c_name(), size);
            throw std::runtime_error("ggml-hex: rpcmem_alloc failed (see log for details)");
        }

        this->fd = rpcmem_to_fd(this->base);
        if (this->fd < 0) {
            GGML_LOG_ERROR("ggml-hex: %s failed to get FD for buffer %p\n", sess->c_name(), (void *) this->base);
            throw std::runtime_error("ggml-hex: rpcmem_to_fd failed (see log for details)");
        }
        this->size = size;

        HEX_VERBOSE("ggml-hex: %s allocated buffer: base %p size %zu fd %d pinned %d\n", sess->c_name(),
                    (void *) this->base, this->size, this->fd, (int) pinned);
        mmap();
    }

    void free() {
        if (!this->base) return;

        unmap();
        rpcmem_free(this->base);

        HEX_VERBOSE("ggml-hex: %s freed buffer: base %p size %zu fd %d\n", sess->c_name(),
                (void *) this->base, size, this->fd);

        this->base = NULL;
    }

    ggml_hexagon_shared_buffer(ggml_hexagon_session * sess, size_t size, bool pinned = false) {
        this->sess   = sess;
        this->size   = 0;
        this->base   = nullptr;
        this->fd     = -1;
        this->mapped = false;
        this->pinned = pinned;

        try {
            alloc(size);
        } catch (...) {
            free();
            throw;
        }
    }

    ~ggml_hexagon_shared_buffer() {
        free();
    }
};
