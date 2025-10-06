import os, csv, time
from typing import Optional

HEADER = ["timestamp","sample_id","pred_label","pred_prob","user_claim","evidence","notes"]

def log_feedback(sample_id: str,
                 pred_label: int,
                 pred_prob: float,
                 user_claim: str,
                 evidence: Optional[str] = None,
                 notes: str = "",
                 path: str = "data/feedback.csv") -> None:
    """追加一条用户反馈记录.

    sample_id: 样本或节点标识（例如节点索引/电话号码）
    pred_label: 模型预测标签
    pred_prob: 预测标签的概率或置信度
    user_claim: 用户主张(如: legit / fraud / unknown)
    evidence: 附加证据信息（文件名/描述/外部ID）
    notes: 额外备注（可空）
    path: CSV 保存路径
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    row = [time.strftime("%Y-%m-%d %H:%M:%S"), sample_id, pred_label, f"{pred_prob:.6f}", user_claim, evidence or "", notes]
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(HEADER)
        w.writerow(row)

if __name__ == "__main__":
    # 简单自测
    log_feedback("node_123", 1, 0.87321, "legit", evidence="waybill:ABC123", notes="user appealed")
    print("feedback appended")
