# Data Schema

## Track Fields

| Field | Units | Allowed Range |
|-------|-------|---------------|
| `storm_id` | string | Basin+number+year (e.g., `AL092019`) |
| `timestamp` | UTC ISO8601 | — |
| `latitude` | degrees north | -90 to 90 |
| `longitude` | degrees east | -180 to 180 |
| `max_wind` | knots | 0 to 200 |
| `min_pressure` | hPa | 800 to 1050 |
| `34kt_ne`, `34kt_se`, `34kt_sw`, `34kt_nw` | nautical miles | 0 to 300 (optional) |

Example (CSV):

```csv
storm_id,timestamp,latitude,longitude,max_wind,min_pressure
AL092019,2019-09-01T00:00:00Z,23.5,-75.0,65,985
```

## ERA5 Variables

| Variable | Units | Pressure Levels (hPa) |
|----------|-------|----------------------|
| `geopotential` | m^2 s^-2 | 1000–50 |
| `temperature` | K | 1000–50 |
| `u_component_of_wind` | m s^-1 | 1000–50 |
| `v_component_of_wind` | m s^-1 | 1000–50 |
| `specific_humidity` | kg kg^-1 | 1000–50 |
| `mean_sea_level_pressure` | Pa | surface |
| `10m_u_component_of_wind` | m s^-1 | surface |
| `10m_v_component_of_wind` | m s^-1 | surface |
| `2m_temperature` | K | surface |

Typical pressure-level values (hPa): `1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50`.

Example (JSON):

```json
{
  "variables": [
    "geopotential",
    "temperature",
    "u_component_of_wind"
  ],
  "pressure_levels": [1000, 850, 500]
}
```
