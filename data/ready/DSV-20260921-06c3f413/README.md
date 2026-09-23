# Training-ready data — DSV-20260921-06c3f413

Only rows that passed every rule are here; everything erased is in `removed.csv`.
The last 20% of days is `split = test`: never fit or choose a model on it.

## train-issued.csv — 285 operations (target: `issued_litres`)

Features: vehicle (+ type, group), activity_mode, distance_km, lifting_hours (non-zero only for Truck Crane 01/02 and Wheel Crane).

| Unit | Group | Train | Test | Mean litres |
|---|---|---|---|---|
| Oil Field Truck | Truck | 56 | 14 | 76.8 |
| Prime Mover | Truck | 35 | 9 | 68.4 |
| Truck Crane 01 | Crane | 46 | 12 | 65.6 |
| Truck Crane 02 | Crane | 44 | 12 | 71.4 |
| Wheel Crane | Crane | 32 | 9 | 111.9 |
| Winch Truck | Truck | 12 | 4 | 76.2 |

## train-measured.csv — 1268 unit-days (target: `measured_litres`)

Features: vehicle (+ type, group), gps_km, gps drive / working / idle hours, lifting_hours (cranes, from the typed operation).

| Unit | Group | Train | Test | Mean litres |
|---|---|---|---|---|
| Prime Mover | Truck | 30 | 8 | 28.2 |
| Truck Crane 01 | Crane | 40 | 10 | 33.8 |
| Truck Crane 02 | Crane | 16 | 4 | 31.0 |
| VT 01 | Vacuum Truck | 100 | 25 | 34.9 |
| VT 02 | Vacuum Truck | 100 | 25 | 20.3 |
| VT 03 | Vacuum Truck | 92 | 24 | 35.4 |
| VT 04 | Vacuum Truck | 98 | 25 | 38.7 |
| VT 05 | Vacuum Truck | 102 | 26 | 44.0 |
| VT 06 | Vacuum Truck | 86 | 22 | 45.1 |
| VT 07 | Vacuum Truck | 83 | 21 | 26.0 |
| VT 08 | Vacuum Truck | 104 | 26 | 53.0 |
| VT 09 | Vacuum Truck | 97 | 25 | 14.2 |
| Wheel Crane | Crane | 30 | 8 | 90.6 |
| Winch Truck | Truck | 32 | 9 | 16.4 |

## Erased

| Table | Rule | Rows |
|---|---|---|
| train-issued | crane_without_lifting_hours | 1 |
| train-issued | outlier_for_unit | 34 |
| train-issued | typed_km_disagrees_with_tracker | 80 |
| train-measured | crane_day_without_typed_hours | 99 |
| train-measured | fs_km_per_litre_implausible | 61 |
| train-measured | fs_negative_consumption | 220 |
| train-measured | fuel_stick_km_disagrees_with_tracker | 92 |
| train-measured | no_consumption_measured | 14 |
| train-measured | no_tracker_record | 206 |
| train-measured | outlier_for_unit | 54 |
| train-measured | typed_km_but_no_movement_recorded | 23 |
| train-measured | unit_km_per_litre_implausible | 44 |
