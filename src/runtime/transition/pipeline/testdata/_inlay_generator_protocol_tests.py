LOG = []


def reset_log():
    LOG.clear()


def completed(value):
    if False:
        yield None
    return value


def exc_name(exc_type):
    if exc_type is None:
        return 'None'
    return exc_type.__name__


def async_method_send_impl():
    def coro():
        received = yield 'method-send-pause'
        return 'method-send-done:' + str(received)

    return coro()


def async_method_throw_impl():
    def coro():
        try:
            yield 'method-throw-pause'
        except ValueError as error:
            return 'method-throw-done:' + str(error)
        return 'method-throw-not-thrown'

    return coro()


def async_method_close_impl():
    def coro():
        try:
            yield 'method-close-pause'
        finally:
            LOG.append('method-close')

    return coro()


class AsyncContextSend:
    def __aenter__(self):
        def coro():
            received = yield 'context-enter-pause'
            LOG.append('context-enter-resume:' + str(received))
            return 'context-entered'

        return coro()

    def __aexit__(self, exc_type, exc_val, exc_tb):
        LOG.append('context-exit:' + exc_name(exc_type))
        return completed(False)


def async_context_send_impl():
    return AsyncContextSend()


class SyncContext:
    def __enter__(self):
        LOG.append('sync-enter')
        return 'sync-entered'

    def __exit__(self, exc_type, exc_val, exc_tb):
        LOG.append('sync-exit:' + exc_name(exc_type))
        return False


def sync_context_impl():
    return SyncContext()


def context_throw_awaitable_impl():
    def coro():
        yield 'context-throw-pause'
        return 'context-throw-not-thrown'

    return coro()


def context_close_awaitable_impl():
    def coro():
        try:
            yield 'context-close-pause'
        finally:
            LOG.append('context-await-close')

    return coro()


class AsyncExitSend:
    def __aexit__(self, exc_type, exc_val, exc_tb):
        LOG.append('async-exit-send-start:' + exc_name(exc_type))

        def coro():
            received = yield 'async-exit-send-pause'
            LOG.append('async-exit-send-resume:' + str(received))
            return True

        return coro()


def async_exit_send_context():
    return AsyncExitSend()


class AsyncExitThrow:
    def __aexit__(self, exc_type, exc_val, exc_tb):
        LOG.append('async-exit-throw-start:' + exc_name(exc_type))

        def coro():
            try:
                yield 'async-exit-throw-pause'
            except ValueError as error:
                LOG.append('async-exit-throw:' + str(error))
                return True
            return False

        return coro()


def async_exit_throw_context():
    return AsyncExitThrow()


class AsyncExitCloseInner:
    def __aexit__(self, exc_type, exc_val, exc_tb):
        LOG.append('async-exit-close-start:' + exc_name(exc_type))

        def coro():
            try:
                yield 'async-exit-close-pause'
            finally:
                LOG.append('async-exit-close-inner')

        return coro()


def async_exit_close_inner_context():
    return AsyncExitCloseInner()


class SyncExitOuter:
    def __exit__(self, exc_type, exc_val, exc_tb):
        LOG.append('sync-exit-outer:' + exc_name(exc_type))
        return False


def sync_exit_outer_context():
    return SyncExitOuter()
