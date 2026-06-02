# Images & Cross-Links — for updating index.html and perception/clustering pages

All images below are already on disk at:
  /home/john/git/RS2-HoloAssist/site/screenshots/

Copy them into this repo's screenshots/ folder before using.

---

## This site — HoloAssist-AI

GitHub repo:   https://github.com/Sebastian-Baudille/HoloAssist-AI
GitHub Pages:  https://sebastian-baudille.github.io/HoloAssist-AI/

**To deploy:** Seb goes to repo Settings → Pages → Source: Deploy from branch → Branch: `site` → / (root) → Save.
The `site` branch already has index.html at root so it will work immediately.

---

## Cross-link to base HoloAssist site

Add a link on this site pointing to the base HoloAssist project site.
Suggested placement: footer "Related Projects" column, hero strip, and/or perception page intro.

Base HoloAssist GitHub repo:  https://github.com/John-A-Chen/HoloAssist
GitHub Pages site URL:        https://john-a-chen.github.io/HoloAssist
  (verify deployed — check repo Settings → Pages)

Suggested footer link text:
  "HoloAssist (base) — Mixed-reality teleoperation + AprilTag perception →"

Suggested perception page intro:
  "This pipeline supersedes the AprilTag 3 system built in the base HoloAssist project.
   See the original system: [HoloAssist →](https://john-a-chen.github.io/HoloAssist)"

---

## Mutual cross-links (add to BOTH sites)

### On HoloAssist-AI (this site) — add to footer and perception page:
  HoloAssist (base): https://john-a-chen.github.io/HoloAssist
  Text: "Base project — Mixed-reality teleoperation, AprilTag perception, Quest 3 dashboard"

### On HoloAssist (base site) — add to footer:
  HoloAssist-AI: https://sebastian-baudille.github.io/HoloAssist-AI/
  Text: "AI extension — DBSCAN point cloud clustering, PPO grasping, Isaac Sim training"

---

## Images available from RS2-HoloAssist

### Perception / camera (use on perception.html and index.html perception slide)

| File | Size | What it shows | Best used for |
|---|---|---|---|
| `perception.png` | 1536×1024 | AprilTag cube tracking in RViz — cubes visible with detection markers | perception slide background, perception.html hero |
| `perception.gif` | 810×540 | Animated — cubes being tracked live as they move | perception slide hover gif (sc-gif) |
| `calibration.png` | 862×783 | Hand-eye calibration in progress — RViz view with arm and tag | calibration section on perception.html |
| `camera_setup.jpeg` | 2194×1646 | Physical RealSense D435i mounted above workspace — real hardware | "sim-to-real" section, shows the real camera rig |
| `multicubetracking.png` | 880×646 | 4 cubes tracked simultaneously in RViz with TF markers | gallery, perception.html — shows accuracy |
| `rviz_cubes.png` | 862×783 | RViz TF tree with calibrated cube poses | calibration verification result |
| `cubebintracking.png` | varies | Cube positions relative to bin in RViz | pick-and-place context |
| `apriltest.png` | varies | AprilTag detection test — tags highlighted on image | perception.html technical detail |
| `aprilcube2docs.png` | varies | AprilTag cube documentation photo — physical printed cube | perception.html, shows what the physical cubes look like |
| `printingaprilcubes.png` | varies | Physical process — printing and attaching AprilTag stickers | perception.html, nice behind-the-scenes |
| `quest_ar_overlay.png` | 1614×924 | Quest 3 AR — cube holograms overlaid on real scene | great for the Unity/XR angle, index.html gallery |

### General / gallery (use in index.html gallery and other slides)

| File | What it shows | Best used for |
|---|---|---|
| `autonomous.png` | Robot performing autonomous pick-and-place | index.html hero background or gallery |
| `autonomous.gif` | Animated autonomous operation | gallery or simulation slide hover |
| `teleoperation.png` | Hand teleoperation with Quest 3 | gallery |
| `teleoperation.gif` | Animated teleoperation | gallery hover or simulation slide |
| `visualisation.png` | RViz + Quest 3 dual visualisation | gallery, architecture section |
| `visualisation.gif` | Animated dual view | gallery hover |
| `dashboard.png` | Steam Deck / web dashboard | Nic's section or gallery |
| `session.png` | Full session — person wearing Quest 3 with robot | hero background or gallery |
| `ollieteleop.png` | Ollie doing teleoperation demo | gallery, team section |

---

## How to copy the images

```bash
cp /home/john/git/RS2-HoloAssist/site/screenshots/perception.png screenshots/
cp /home/john/git/RS2-HoloAssist/site/screenshots/perception.gif screenshots/
cp /home/john/git/RS2-HoloAssist/site/screenshots/calibration.png screenshots/
cp /home/john/git/RS2-HoloAssist/site/screenshots/camera_setup.jpeg screenshots/
cp /home/john/git/RS2-HoloAssist/site/screenshots/multicubetracking.png screenshots/
cp /home/john/git/RS2-HoloAssist/site/screenshots/rviz_cubes.png screenshots/
cp /home/john/git/RS2-HoloAssist/site/screenshots/quest_ar_overlay.png screenshots/
cp /home/john/git/RS2-HoloAssist/site/screenshots/cubebintracking.png screenshots/
cp /home/john/git/RS2-HoloAssist/site/screenshots/apriltest.png screenshots/
cp /home/john/git/RS2-HoloAssist/site/screenshots/aprilcube2docs.png screenshots/
cp /home/john/git/RS2-HoloAssist/site/screenshots/printingaprilcubes.png screenshots/
cp /home/john/git/RS2-HoloAssist/site/screenshots/autonomous.png screenshots/
cp /home/john/git/RS2-HoloAssist/site/screenshots/autonomous.gif screenshots/
cp /home/john/git/RS2-HoloAssist/site/screenshots/teleoperation.png screenshots/
cp /home/john/git/RS2-HoloAssist/site/screenshots/teleoperation.gif screenshots/
cp /home/john/git/RS2-HoloAssist/site/screenshots/visualisation.png screenshots/
cp /home/john/git/RS2-HoloAssist/site/screenshots/visualisation.gif screenshots/
cp /home/john/git/RS2-HoloAssist/site/screenshots/session.png screenshots/
cp /home/john/git/RS2-HoloAssist/site/screenshots/ollieteleop.png screenshots/
cp /home/john/git/RS2-HoloAssist/site/screenshots/dashboard.png screenshots/
cp /home/john/git/RS2-HoloAssist/site/screenshots/quest_ar_overlay.png screenshots/
```

---

## Recommended gallery slots (index.html)

Replace the 6 placeholder gallery items:

```
Slot 1: perception.png       → "AprilTag perception — 20 Hz, 4 cubes tracked"
Slot 2: multicubetracking.png → "Multi-cube tracking in RViz"
Slot 3: calibration.png      → "Hand-eye calibration — Park solver, <2mm error"
Slot 4: quest_ar_overlay.png → "Quest 3 AR — cube holograms on real scene"
Slot 5: session.png          → "Full system demo"
Slot 6: autonomous.gif       → "Autonomous pick-and-place episode"  ← keep as video-style
```

---

## Recommended perception slide update (index.html 01/04)

Change the slide background from `.s-perc { background-image: url('screenshots/perception.png'); }`
Add hover gif: `<img class="sc-gif" src="screenshots/perception.gif" alt="">`

Suggested tag line update:
  "AprilTag 3 (real hardware) → DBSCAN point cloud (sim) · ROS 2 · 20 Hz · <2mm calibration"

---

## Perception.html additions

Add a new section "Original System — AprilTag" with:
- Hero image: `perception.png`
- Photo row: `camera_setup.jpeg` + `calibration.png`
- Photo row: `multicubetracking.png` + `rviz_cubes.png`
- Photo row: `aprilcube2docs.png` + `printingaprilcubes.png`
- Cross-link box: "Full documentation for the AprilTag system: [HoloAssist →](https://john-a-chen.github.io/HoloAssist/docs/perception.html)"

Add calibration section:
- Image: `calibration.png`
- Stat: <2mm reprojection error, 12-15 samples, easy_handeye2 (Park solver)
- Cross-link: "[Calibration guide →](https://john-a-chen.github.io/HoloAssist/docs/calibration.html)"
