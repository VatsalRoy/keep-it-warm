Change vs P0 (total cost per run of 12 sessions).

| Policy | Input tokens | Naive cost A | True cost A | Naive cost B | True cost B |
|---|---|---|---|---|---|
| P0 full | +0% | +0% | +0% | +0% | +0% |
| P1 window | -46% | -42% | -14% | -42% | +96% |
| P2 mask | -23% | -21% | +1% | -21% | +87% |
| P3 batch | -22% | -19% | -11% | -19% | +22% |
| P4 summary | -26% | -23% | -20% | -23% | -5% |
