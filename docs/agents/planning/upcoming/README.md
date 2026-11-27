# Upcoming Work

Work that is **understood but not scheduled**. Everything here has had enough analysis that a
future session can pick it up without re-deriving the diagnosis — and enough open questions
that starting it today would be premature.

Nothing here is in flight. An entry earning a schedule graduates to its own `PLAN.md` under the
relevant project folder (e.g. `../bow_valley/`).

| Doc                                                        | What it covers                                                                          |
| ---------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| [TASK-generalize-grid-crs.md](TASK-generalize-grid-crs.md) | Generalize the grid CRS beyond UTM 11N so a second region is expressible.               |
| [potential_future_work.md](potential_future_work.md)       | Known-suboptimal designs not currently paying rent in bugs (diagnosis + fix + trigger). |

## Why a separate folder

Deferred work kept inside a project's task tree reads as "part of the plan, just not done yet",
which is how a parked design decision quietly becomes an assumed one. These are parked
deliberately. Moving them out keeps `../bow_valley/020-data-ingestion/tasks/` a list of work
that was actually specified and built.
