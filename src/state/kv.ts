import type { ISdk } from 'iii-sdk'
import { isInprocEngine } from '../config.js'

const SET_MANY_CHUNK = 100

export class StateKV {
  constructor(private sdk: ISdk) {}

  async get<T = unknown>(scope: string, key: string): Promise<T | null> {
    return this.sdk.trigger<{ scope: string; key: string }, T | null>({
      function_id: 'state::get',
      payload: { scope, key },
    })
  }

  async set<T = unknown>(scope: string, key: string, value: T): Promise<T> {
    return this.sdk.trigger<{ scope: string; key: string; value: T }, T>({
      function_id: 'state::set',
      payload: { scope, key, value },
    })
  }

  async update<T = unknown>(
    scope: string,
    key: string,
    ops: Array<{ type: string; path: string; value?: unknown }>,
  ): Promise<T> {
    return this.sdk.trigger<
      { scope: string; key: string; ops: Array<{ type: string; path: string; value?: unknown }> },
      T
    >({
      function_id: 'state::update',
      payload: { scope, key, ops },
    })
  }

  async delete(scope: string, key: string): Promise<void> {
    return this.sdk.trigger<{ scope: string; key: string }, void>({
      function_id: 'state::delete',
      payload: { scope, key },
    })
  }

  // Bounded batches. Under inproc each chunk is one `state::set-many` call -
  // one transaction, one fsync - and the await between chunks lets the event
  // loop turn; a `state::set` per row was 45k fsyncs back to back (day-0 soak
  // finding). Under iii that function does not exist, so the rows go as
  // ordered single sets, each an RPC that yields on its own.
  async setMany<T = unknown>(scope: string, entries: Array<{ key: string; value: T }>): Promise<number> {
    if (!isInprocEngine()) {
      for (const e of entries) await this.set(scope, e.key, e.value)
      return entries.length
    }
    for (let i = 0; i < entries.length; i += SET_MANY_CHUNK) {
      await this.sdk.trigger<{ scope: string; entries: Array<{ key: string; value: T }> }, number>({
        function_id: 'state::set-many',
        payload: { scope, entries: entries.slice(i, i + SET_MANY_CHUNK) },
      })
    }
    return entries.length
  }

  async list<T = unknown>(scope: string): Promise<T[]> {
    return this.sdk.trigger<{ scope: string }, T[]>({
      function_id: 'state::list',
      payload: { scope },
    })
  }
}
