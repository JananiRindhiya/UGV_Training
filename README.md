# UGV Adaptive Training Simulator — Step 7 Scroll-Fixed Analytics

This version fixes the analytics dashboard scrolling. Open the dashboard with **P**.

- Mouse wheel while dashboard is open: scroll analytics content.
- P: close dashboard.
- The top dashboard header stays fixed.
- Recent Training Runs now has a larger table with multiple records.

Controls in simulator: W/S drive, A/D steer, M autopilot, T adaptive AI, 1/2/3 difficulty, N new terrain, R reset, P analytics, C camera follow, F reset camera, mouse drag orbit/pan, wheel zoom.


## Terrain obstacle update
This version adds typed terrain hazards: ROCK, BOULDER, POTHOLE and LOG.
- Rocks, boulders and logs are solid obstacles and trigger collision/replanning.
- Potholes are drive-through terrain hazards that slow the UGV and are recorded as terrain hits.
- Obstacle types are generated deterministically from the terrain seed and increase in variety/count with difficulty.
- Training CSV analytics now include terrain hazard-hit counts.

### Controls
W/Up = forward, S/Down = reverse, A/D or Left/Right = steering, M = manual/autopilot,
1/2/3 = difficulty, N = new terrain seed, P = analytics dashboard, R = reset, F = camera reset.
