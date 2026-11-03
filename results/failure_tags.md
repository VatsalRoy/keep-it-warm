Questions answered correctly under P0 but not under the policy.
Tags are automatic and can overlap; refine them by hand in failures.csv.

| Policy | Lost vs P0 | after_summary | answered_fast | context_edited_during_q | format_slip | gave_up | other | repeated_calls | turn_limit | wrong_order |
|---|---|---|---|---|---|---|---|---|---|---|
| P1 window | 5 | 0 | 1 | 5 | 1 | 0 | 0 | 0 | 0 | 0 |
| P2 mask | 8 | 0 | 1 | 8 | 1 | 1 | 0 | 2 | 1 | 1 |
| P3 batch | 6 | 0 | 2 | 4 | 1 | 1 | 1 | 0 | 0 | 0 |
| P4 summary | 9 | 5 | 2 | 0 | 1 | 3 | 0 | 2 | 0 | 0 |

| Policy | Extra turns per question vs P0 | Questions that took longer |
|---|---|---|
| P1 window | +0.40 | 11/48 |
| P2 mask | +1.27 | 27/48 |
| P3 batch | +0.60 | 20/48 |
| P4 summary | +0.22 | 13/96 |
