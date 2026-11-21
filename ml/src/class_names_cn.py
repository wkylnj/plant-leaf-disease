"""38 个类别的中文名映射。

用法：
    from class_names_cn import CLASS_CN, to_cn
    to_cn("15_Orange_Haunglongbing_Citrus_greening")  # -> "柑橘黄龙病"

注意：
- 键是数据集目录名（带两位索引前缀），与 ml/data/processed/plantvillage/*/ 下的目录名一致。
- 数据集原始拼写有误（Haunglongbing 应为 Huanglongbing），这里保留原拼写做键，
  中文名用正确写法。报告中若引用英文类名，可注明"数据集原始拼写如此"。
- 中文名采用国内植保文献常用译名，个别病害有多种叫法时取最常见的。
"""

# 数据集目录名 -> (中文病名, 作物)
CLASS_CN = {
    "00_Apple_Apple_scab": ("苹果黑星病", "苹果"),
    "01_Apple_Black_rot": ("苹果黑腐病", "苹果"),
    "02_Apple_Cedar_apple_rust": ("苹果雪松锈病", "苹果"),
    "03_Apple_healthy": ("苹果健康", "苹果"),
    "04_Blueberry_healthy": ("蓝莓健康", "蓝莓"),
    "05_Cherry_including_sour_Powdery_mildew": ("樱桃白粉病", "樱桃"),
    "06_Cherry_including_sour_healthy": ("樱桃健康", "樱桃"),
    "07_Corn_maize_Cercospora_leaf_spot_Gray_leaf_spot": ("玉米灰斑病", "玉米"),
    "08_Corn_maize_Common_rust": ("玉米普通锈病", "玉米"),
    "09_Corn_maize_Northern_Leaf_Blight": ("玉米北方叶枯病", "玉米"),
    "10_Corn_maize_healthy": ("玉米健康", "玉米"),
    "11_Grape_Black_rot": ("葡萄黑腐病", "葡萄"),
    "12_Grape_Esca_Black_Measles": ("葡萄埃斯卡病", "葡萄"),
    "13_Grape_Leaf_blight_Isariopsis_Leaf_Spot": ("葡萄叶斑病", "葡萄"),
    "14_Grape_healthy": ("葡萄健康", "葡萄"),
    "15_Orange_Haunglongbing_Citrus_greening": ("柑橘黄龙病", "柑橘"),
    "16_Peach_Bacterial_spot": ("桃细菌性穿孔病", "桃"),
    "17_Peach_healthy": ("桃健康", "桃"),
    "18_Pepper_bell_Bacterial_spot": ("甜椒细菌性斑点病", "甜椒"),
    "19_Pepper_bell_healthy": ("甜椒健康", "甜椒"),
    "20_Potato_Early_blight": ("马铃薯早疫病", "马铃薯"),
    "21_Potato_Late_blight": ("马铃薯晚疫病", "马铃薯"),
    "22_Potato_healthy": ("马铃薯健康", "马铃薯"),
    "23_Raspberry_healthy": ("树莓健康", "树莓"),
    "24_Soybean_healthy": ("大豆健康", "大豆"),
    "25_Squash_Powdery_mildew": ("南瓜白粉病", "南瓜"),
    "26_Strawberry_Leaf_scorch": ("草莓叶焦病", "草莓"),
    "27_Strawberry_healthy": ("草莓健康", "草莓"),
    "28_Tomato_Bacterial_spot": ("番茄细菌性斑点病", "番茄"),
    "29_Tomato_Early_blight": ("番茄早疫病", "番茄"),
    "30_Tomato_Late_blight": ("番茄晚疫病", "番茄"),
    "31_Tomato_Leaf_Mold": ("番茄叶霉病", "番茄"),
    "32_Tomato_Septoria_leaf_spot": ("番茄斑枯病", "番茄"),
    "33_Tomato_Spider_mites_Two_spotted_spider_mite": ("番茄二斑叶螨", "番茄"),
    "34_Tomato_Target_Spot": ("番茄靶斑病", "番茄"),
    "35_Tomato_Tomato_Yellow_Leaf_Curl_Virus": ("番茄黄化曲叶病毒病", "番茄"),
    "36_Tomato_Tomato_mosaic_virus": ("番茄花叶病毒病", "番茄"),
    "37_Tomato_healthy": ("番茄健康", "番茄"),
}

# 中文名 -> 数据集目录名（反馈接口需要反向查）
CN_TO_CLASS = {cn: key for key, (cn, _crop) in CLASS_CN.items()}


def to_cn(label: str) -> str:
    """英文/目录名 -> 中文病名。找不到时返回原字符串，不抛异常。

    兼容两种输入：
      - 带索引前缀的目录名 "15_Orange_Haunglongbing_Citrus_greening"
      - 不带前缀的原始名 "Orange___Haunglongbing_(Citrus_greening)"
    """
    if label in CLASS_CN:
        return CLASS_CN[label][0]

    # 尝试去掉索引前缀后按归一化形式匹配
    norm = _normalize(label)
    for key, (cn, _crop) in CLASS_CN.items():
        if _normalize(key[3:]) == norm or _normalize(key) == norm:
            return cn
    return label


def crop_of(label: str) -> str:
    """取作物名，找不到返回空串。"""
    if label in CLASS_CN:
        return CLASS_CN[label][1]
    norm = _normalize(label)
    for key, (_cn, crop) in CLASS_CN.items():
        if _normalize(key[3:]) == norm or _normalize(key) == norm:
            return crop
    return ""


def _normalize(s: str) -> str:
    """把 PlantVillage 原始的 '___' 分隔、括号、空格、连字符统一成下划线形式。

    例：'Tomato___Spider_mites Two-spotted_spider_mite'
        -> 'tomato_spider_mites_two_spotted_spider_mite'
    """
    out = s.replace("___", "_").replace("(", "").replace(")", "")
    out = out.replace(",", "").replace(" ", "_").replace("-", "_")
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_").lower()


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    # 默认读本地转换好的类别清单，校验是否 38 个都能映射
    default = Path(__file__).resolve().parents[1] / "data" / "processed" / "plantvillage" / "classes.json"
    if len(sys.argv) > 1:
        default = Path(sys.argv[1])

    if default.exists():
        raw = json.load(open(default, encoding="utf-8"))
        # classes.json 可能是 ["00_Apple_Apple_scab", ...] 也可能是
        # [{"index":0,"folder":"00_Apple_Apple_scab","name":"Apple__Apple_scab"}, ...]
        if raw and isinstance(raw[0], dict):
            classes = [r.get("folder") or r.get("name") for r in raw]
            originals = [r.get("name") for r in raw]
        else:
            classes = list(raw)
            originals = list(raw)

        missing = [c for c in classes if c not in CLASS_CN]
        print(f"classes.json 共 {len(classes)} 类")
        print(f"映射表共 {len(CLASS_CN)} 条")
        print(f"按目录名未覆盖: {missing if missing else '无'}")

        # 原始名（Apple___Apple_scab 形式）也要能反查
        miss_orig = [o for o in originals if o and o not in CLASS_CN and to_cn(o) == o]
        print(f"按原始名未覆盖: {miss_orig if miss_orig else '无'}")

        print("\n前 5 条：")
        for c in classes[:5]:
            print(f"  {c} -> {to_cn(c)}（{crop_of(c)}）")
    else:
        print(f"未找到 {default}，仅打印映射表前 5 条：")
        for k in list(CLASS_CN)[:5]:
            print(f"  {k} -> {CLASS_CN[k][0]}")
