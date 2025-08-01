# PMH-PCD

## Compact-yet-Separate: Proto-centric Multi-modal Hashing with Pronounced Category Differences for Multi-modal Retrieval

## Overview
**PMH-PCD (Proto-centric Multi-modal Hashing with Pronounced Category Differences)** is a novel framework designed to address the challenges of balancing intraclass compactness and interclass separability in multi-modal hashing tasks. PMH-PCD uses compact hash codes to represent multi-modal data, achieving low storage costs and high retrieval speeds. 

### Key Contributions:
1. **Modality-Specific Prototypes:** PMH-PCD deeply explores within-modality class information to ensure the effective fusion of unique characteristics from each modality.
2. **Multi-modal Integrated Class Prototypes:** By incorporating semantic information across modalities, PMH-PCD captures intricate relationships and complementary semantic content within multi-modal data.
3. **Discriminative Binary Hash Codes:** It holistically integrates multifaceted semantic information, including low-level pairwise relations and high-level structural patterns, to generate more representative and discriminative hash codes.
4. **Superior and Consistent Performance:** Experimental results show that PMH-PCD outperforms state-of-the-art methods across multiple datasets.

---

## Abstract
Multi-modal hashing achieves low storage costs and high retrieval speeds by using compact hash codes to represent complex and heterogeneous multi-modal data, effectively addressing the inefficiency and resource intensiveness challenges faced by traditional multi-modal retrieval methods. However, balancing intraclass compactness and interclass separability remains a struggle in existing works due to coarse-grained feature limitations, simplified fusion strategies that overlook semantic complementarity, and neglect of the structural information within the multi-modal data. To address these limitations comprehensively, we propose a **Proto-centric Multi-modal Hashing with Pronounced Category Differences (PMH-PCD)** model. Specifically, PMH-PCD first learns modality-specific prototypes by deeply exploring within-modality class information, ensuring effective fusion of each modality's unique characteristics. Furthermore, it learns multi-modal integrated class prototypes that seamlessly incorporate semantic information across modalities to effectively capture and represent the intricate relationships and complementary semantic content embedded within the multi-modal data. Additionally, to generate more discriminative and representative binary hash codes, PMH-PCD integrates multifaceted semantic information, encompassing both low-level pairwise relations and high-level structural patterns, holistically capturing intricate data details and leveraging underlying structures. The experimental results demonstrate that, compared with existing advanced methods, PMH-PCD achieves superior and consistent performance in multi-modal retrieval tasks.

---

## Authors
This work was authored by:
- **Ruifan Zuo**
- **Chaoqun Zheng**
- **Lei Zhu**
- **Wenpeng Lu**
- **Jiasheng Si**
- **Weiyu Zhang**

Published in **IEEE Transactions on Multimedia**.

---

## Datasets
PMH-PCD is evaluated on three benchmark datasets: **MIRFlickr**, **MS COCO**, and **NUS-WIDE**. The details of these datasets are as follows:

| Dataset    | Categories | Training Samples | Retrieval Samples | Query Samples |
|------------|------------|------------------|-------------------|---------------|
| MIRFlickr  | 24         | 5,000            | 17,772            | 2,243         |
| MS COCO    | 80         | 18,000           | 82,783            | 5,981         |
| NUS-WIDE   | 21         | 21,000           | 193,749           | 2,085         |

---

## Experimental Setup

### Hardware:
- **GPU:** NVIDIA RTX 3090  

### Software:
- **Python Version:** 3.8.18  
- **PyTorch Version:** 1.10.1  

---

## Usage
To train the model on a specific dataset such as **MIRFlickr**, execute the following command:
```bash
bash flickr.sh
```

## Results
PMH-PCD achieves superior performance compared to state-of-the-art methods, demonstrating its effectiveness in multi-modal retrieval tasks.

## Citation
If you find this work helpful, please consider citing our paper:
```bash
@ARTICLE{zuo2025,
  author={Zuo, Ruifan and Zheng, Chaoqun and Zhu, Lei and Lu, Wenpeng and Si, Jiasheng and Zhang, Weiyu},
  journal={IEEE Transactions on Multimedia}, 
  title={Compact-yet-Separate: Proto-centric Multi-modal Hashing with Pronounced Category Differences for Multi-modal Retrieval}, 
  year={2025},
  pages={1-14}}
}
```

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Contact
For any questions or issues, please feel free to contact:

Ruifan Zuo: [zrfan9928@gmail.com]



