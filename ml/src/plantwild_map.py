"""PlantWild → PlantVillage 类别映射表。

为什么需要这个文件
------------------
PlantDoc 只覆盖我们 38 类中的 27 类（2920 张），且「健康」类极不友好。
PlantWild（Wei et al., ACM MM 2024）是当前最大的田间植物病害分类数据集：
  - v1: 18542 张 / 89 类（其中 30 类是「干净叶片」即健康类）
  - v2: 11488 张 / 115 类（农业专家精修，但去掉了健康类，只剩病害）

对我们的价值：v1 的 89 类里，有 33 类能对上我们 PlantVillage 的 38 类，
其中包含 13 个「健康」类（apple leaf / corn leaf / tomato leaf …），
正好补上 PlantDoc 最缺的那一块（PlantDoc 上健康类几乎全错）。

命名约定
--------
PlantWild v1 的目录名是「作物 + 病名」的小写英文，健康类是「<作物> leaf」。
本表把 PlantVillage 的文件夹名（如 `00_Apple_Apple_scab`）映射到
PlantWild v1 的目录名；映射为 None 表示 PlantWild 没有对应类别。
"""
from __future__ import annotations

# PlantVillage 文件夹名 → PlantWild v1 目录名（None = 无对应）
PLANTWILD_TO_PV: dict[str, str | None] = {
    # ---- Apple ----
    "00_Apple_Apple_scab": "apple scab",
    "01_Apple_Black_rot": "apple black rot",
    "02_Apple_Cedar_apple_rust": "apple rust",
    "03_Apple_healthy": "apple leaf",
    # ---- Blueberry ----
    "04_Blueberry_healthy": "blueberry leaf",
    # ---- Cherry ----
    "05_Cherry_including_sour_Powdery_mildew": "cherry powdery mildew",
    "06_Cherry_including_sour_healthy": "cherry leaf",
    # ---- Corn ----
    "07_Corn_maize_Cercospora_leaf_spot_Gray_leaf_spot": "corn gray leaf spot",
    "08_Corn_maize_Common_rust": "corn rust",
    "09_Corn_maize_Northern_Leaf_Blight": "corn northern leaf blight",
    "10_Corn_maize_healthy": "corn leaf",
    # ---- Grape ----
    "11_Grape_Black_rot": "grape black rot",
    # Esca（黑麻疹）在 PlantWild 没有独立类别，最容易和 leaf spot 混，故不映射
    "12_Grape_Esca_Black_Measles": None,
    # Isariopsis 叶斑与 grape leaf spot 病名不完全一致，为避免标签噪声不映射
    "13_Grape_Leaf_blight_Isariopsis_Leaf_Spot": None,
    "14_Grape_healthy": "grape leaf",
    # ---- Orange ----
    "15_Orange_Haunglongbing_Citrus_greening": "citrus greening disease",
    # ---- Peach ----
    # PlantWild 只有 peach leaf curl / peach scab 等，没有 bacterial spot
    "16_Peach_Bacterial_spot": None,
    "17_Peach_healthy": "peach leaf",
    # ---- Pepper bell ----
    # PlantWild 是 "bell pepper leaf spot"，语义接近 bacterial spot，但不确定，保守不映射
    "18_Pepper_bell_Bacterial_spot": None,
    "19_Pepper_bell_healthy": "bell pepper leaf",
    # ---- Potato ----
    "20_Potato_Early_blight": "potato early blight",
    "21_Potato_Late_blight": "potato late blight",
    "22_Potato_healthy": "potato leaf",
    # ---- Raspberry ----
    "23_Raspberry_healthy": "raspberry leaf",
    # ---- Soybean ----
    "24_Soybean_healthy": "soybean leaf",
    # ---- Squash ----
    "25_Squash_Powdery_mildew": "squash powdery mildew",
    # ---- Strawberry ----
    "26_Strawberry_Leaf_scorch": "strawberry leaf scorch",
    "27_Strawberry_healthy": "strawberry leaf",
    # ---- Tomato ----
    "28_Tomato_Bacterial_spot": "tomato bacterial leaf spot",
    "29_Tomato_Early_blight": "tomato early blight",
    "30_Tomato_Late_blight": "tomato late blight",
    "31_Tomato_Leaf_Mold": "tomato leaf mold",
    "32_Tomato_Septoria_leaf_spot": "tomato septoria leaf spot",
    # 红蜘蛛不是病害（虫害），PlantWild 无对应
    "33_Tomato_Spider_mites_Two_spotted_spider_mite": None,
    # Target Spot 在 PlantWild 无独立类别
    "34_Tomato_Target_Spot": None,
    "35_Tomato_Tomato_Yellow_Leaf_Curl_Virus": "tomato yellow leaf curl virus",
    "36_Tomato_Tomato_mosaic_virus": "tomato mosaic virus",
    "37_Tomato_healthy": "tomato leaf",
}

# 反向：PlantWild 目录名 → PlantVillage 文件夹名
PV_TO_PLANTWILD: dict[str, str] = {
    pw: pv for pv, pw in PLANTWILD_TO_PV.items() if pw is not None
}

# 保守映射之外、语义上可能对应但不采用的类别（写文档时用）
REJECTED: dict[str, str] = {
    "13_Grape_Leaf_blight_Isariopsis_Leaf_Spot": "grape leaf spot",
    "16_Peach_Bacterial_spot": "peach leaf curl / peach scab",
    "18_Pepper_bell_Bacterial_spot": "bell pepper leaf spot",
    "34_Tomato_Target_Spot": "tomato anthracnose",
}


def mapped_classes() -> list[str]:
    """返回能映射的 PlantVillage 文件夹名（保持 38 类的顺序）。"""
    return [pv for pv, pw in PLANTWILD_TO_PV.items() if pw is not None]


def unmapped_classes() -> list[str]:
    """返回 PlantWild 无对应的 PlantVillage 类别。"""
    return [pv for pv, pw in PLANTWILD_TO_PV.items() if pw is None]


if __name__ == "__main__":
    m, u = mapped_classes(), unmapped_classes()
    print(f"可映射 {len(m)}/38 类，不可映射 {len(u)} 类：")
    for c in u:
        print(f"   {c}  →  PlantWild 候选：{REJECTED.get(c, '无')}")
