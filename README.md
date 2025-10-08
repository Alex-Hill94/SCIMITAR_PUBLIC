# SCIMITAR_PUBLIC


**SCIMITAR: Optimising chest digital tomosynthesis devices using geometric simulations and genetic algorithms**

This repository contains the publicly available version of the SCIMITAR software developed for the paper by A.D. Hill et al. (DOI 10.1088/2057-1976/ae0fa0).

## About

SCIMITAR is a geometry-based simulation framework designed to optimise chest digital tomosynthesis (DT) systems. The software models X-ray radiation coverage in multi-panel flat panel source (FPS) configurations and uses genetic algorithms to identify optimal device parameters.

### Key Features

- **Geometric simulation** of X-ray radiation coverage in chest DT systems
- **Performance evaluation** using irradiation uniformity metrics

### Applications

SCIMITAR was developed to design mobile chest DT devices for clinical applications including:
- Nasogastric tube placement verification
- Early lung cancer detection
- Low-dose 3D chest imaging

The framework enables rapid exploration of large design spaces to identify configurations that balance imaging performance, engineering feasibility, and clinical requirements.

## 🚧 Repository Status

**This repository is currently under active development and is subject to change.**

Package versions used by SCIMITAR:

``` 
Pillow==10.2.0
matplotlib==3.8.0
numpy==1.26.3
pandas==1.3.4
scipy==1.11.4
tqdm==4.65.0
vtk==9.2.6
```



Content to be added includes:
- Source code
- Documentation
- Example configurations
- Usage instructions
- Installation guide


## Citation

If you use this software in your research, please cite both the software and the source paper:

### Software
  
  @software{hill2025scimitar_software,  
  author = {Hill, Alexander and Aflyatunova, Daliya and Holloway, Fraser},  
  title = {SCIMITAR: Publicly Available Software},  
  year = {2025},  
  publisher = {GitHub},  
  url = {https://github.com/Alex-Hill94/SCIMITAR_PUBLIC}  
  }

### Paper

  @article{hill2025scimitar,  
  title = {SCIMITAR: Optimising chest digital tomosynthesis devices using geometric simulations and genetic algorithms},  
  author = {Hill, A.D. and others},  
  journal = {TBD},  
  year = {2025},  
  note = {In press}  
  }  

Full paper reference will be added upon publication.

## Contact

For questions or inquiries about this software, please contact a.d.hill@liverpool.ac.uk.

## Acknowledgments

This work was developed in collaboration with Adaptix Ltd., pioneers in cold-cathode X-ray emitter array technology for flat panel source imaging systems.

---

