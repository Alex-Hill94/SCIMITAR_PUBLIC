# **SCIMITAR_PUBLIC**

**Simulating Complete Irradiation Maps and Improving Tomosynthesis in X-ray Radiography**

---

This repository contains the publicly available version of the **SCIMITAR** software developed for the paper:

> *Hill, A.D. et al. (2025)* — **SCIMITAR: Optimising chest digital tomosynthesis devices using geometric simulations and genetic algorithms**  
> DOI: *10.1088/2057-1976/ae0fa0*

---

## 🧠 Overview

**SCIMITAR** is a Python-based framework for designing and analysing **chest digital tomosynthesis (DT)** geometries. It models X-ray radiation coverage in multi-panel flat panel source (FPS) configurations by simulating intersections between distributed X-ray emitters and a detector plane.

It generates irradiation maps, computes quantitative metrics, and supports interactive 3D visualisation using **VTK** — enabling rapid exploration of large design spaces to balance imaging performance, engineering feasibility, and clinical requirements.

---

## ✨ Key Features

- **Flexible DT device design** – Create single- or multi-panel configurations with custom emitter grids, cone angles, and panel orientations.  
- **Irradiation mapping** – Compute pixel-wise irradiation distributions on arbitrary intersection planes.  
- **Quantitative metrics** – Assess coverage, emitter overlap, gantry angle, and stray radiation.  
- **3D visualisation** – View panels, cones, and patient models in an interactive VTK scene.

---

## 📁 Repository Structure

| File | Description |
|------|--------------|
| `Run.py` | Example execution script demonstrating a full SCIMITAR workflow |
| `SCIMITAR.py` | Main module defining the `Scimitar` class and core methods |
| `aux_material.py` | Helper classes and utility functions supporting the main class |
| `clipping_utils.py` | VTK-based functions for geometric intersection calculations |
| `read_patient.py` | Utilities for loading an illustrative patient mesh |
| `interactor_utils.py` | VTK scene interaction and visualisation utilities |

---

## 🧩 Dependencies

``` 
Pillow==10.2.0
matplotlib==3.8.0
numpy==1.26.3
pandas==1.3.4
scipy==1.11.4
tqdm==4.65.0
vtk==9.2.6
```


---

## 🚀 Installation & Example Run

Clone and navigate to the repository:

```
git clone https://github.com/Alex-Hill94/SCIMITAR_PUBLIC.git
cd SCIMITAR_PUBLIC
```

Run the example workflow:

```
python Run.py
```

This builds a 4-panel system (each with a 4×4 emitter grid), computes and plots irradiation maps at detector and mid-height planes, visualises the geometry, and prints performance metrics.

---

### Example Console Output

```
Acceptable geometry, creating visualisation...
Reading SCIMITAR_PUBLIC/patient/skeletal: 100%|██████| 49/49 [00:00<00:00, 138.91it/s]
Reading SCIMITAR_PUBLIC/patient/skull: 100%|██████| 9/9 [00:00<00:00, 76.63it/s]
                               0
Cone_Angle             31.900000
SID                     0.976000
Emitter_Pitch           0.010000
Panel_Pitch             0.229000
Panel_Angle            12.500000
Angular_Range          27.329982
Illumination          100.000000
Overlap_2D             86.512980
Overlap_Mid_2D         63.912926
Overlap_3D             63.912926
Stray_Radiation       808.000000
Panel_Collision         0.000000
Panel_Xray_Collision    0.000000
Outside_Wafer           0.000000
Acceptable_Stray        1.000000
Contained_Panels        1.000000
```

---

### Example Visual Outputs

<p align="center">
  <img src="example_irmap.png.png" alt="Example irradiation map computed by SCIMITAR" width="45%"/>
  <img src="example_vtk_view.png" alt="Example VTK visualisation of SCIMITAR geometry" width="45%"/>
</p>

---

## 📊 Selected Output Metrics

| Metric | Description |
|---------|--------------|
| **Overlap_2D** | Uniformity of plane (e.g. detector surface) irradiation by all emitters |
| **Overlap_3D** | 3D extension of 2D overlap; evaluates volumetric coverage (requires `intersection_heights = full_intersection_heights`) |
| **Angular Range** | Maximum angular span between any pair of emitters |
| **Stray Radiation** | Number of detector pixels irradiated outside the detector bounds |

---

## 🔄 Typical Workflow

1. Modify input parameters in `Run.py` (e.g. cone angle, emitter count, panel tilt).  
2. Generate irradiation maps with `S.Irradiation()`.  
3. Evaluate metrics via `S.Metrics()`.  
4. Visualise geometry using `S.Visualise()`.  
5. Export results from the DataFrame `S.df`.

---

## 📚 Citation

If you use **SCIMITAR** in your research, please cite both the software and the associated paper:

### Software
```
  @software{hill2025scimitar_software,  
  author = {Hill, Alexander and Aflyatunova, Daliya and Holloway, Fraser},  
  title = {SCIMITAR: Publicly Available Software},  
  year = {2025},  
  publisher = {GitHub},  
  url = {https://github.com/Alex-Hill94/SCIMITAR_PUBLIC}  
  }

```

### Paper
```  
@article{hill2025scimitar,  
  title = {SCIMITAR: Optimising chest digital tomosynthesis devices using geometric simulations and genetic algorithms},  
  author = {Hill, A.D. and others},  
  journal = {TBD},  
  year = {2025},  
  note = {In press}  
  }  
```
> The paper is currently available as an accepted manuscript. Full citation details will be added upon publication.

---

## 📬 Contact

For questions or inquiries, please contact  
**a.d.hill@liverpool.ac.uk**

---

## 🤝 Acknowledgments

Developed in collaboration with **Adaptix Ltd.**, pioneers in novel X-ray imaging technologies.

---

### 🔗 Useful Links
- [Adaptix Ltd.](https://www.adaptix.com)
- [VTK Documentation](https://vtk.org/documentation/)
- [QUASAR Group](https://www.youtube.com/watch?v=ezp9lAFdRdU)