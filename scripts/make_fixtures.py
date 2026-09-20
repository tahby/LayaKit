import json
import sys
from pathlib import Path

import numpy as np

from laya_coreml.agent import Agent
from laya_coreml.ane import ANEAgent
from laya_coreml.inputs import collate_items

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = REPO_ROOT / "Tests" / "LayaKitTests" / "Fixtures" / "fixtures.json"
ANE_MAX_LENGTH = 96


def to_plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {key: to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(item) for item in value]
    return value


shipment_sentence = (
    "The customer wrote in about a delayed shipment and asked for a refund or "
    "replacement as soon as possible please. "
)
shipment_words = (shipment_sentence * 10).split()
state_70_tokens = " ".join(shipment_words[:67])
state_90_tokens = " ".join(shipment_words[:86])

CASES = [
    {
        "id": "choice_yesno_short",
        "state": "I was billed twice. Refund the duplicate.",
        "question": {
            "type": "choice",
            "instructions": "Who should handle this?",
            "criteria": ["billing", "technical"],
        },
    },
    {
        "id": "choice_k3_routing",
        "state": "The customer says the app crashes every time they open settings.",
        "question": {
            "type": "choice",
            "instructions": "Which team should own this ticket?",
            "criteria": ["billing", "technical", "sales"],
        },
    },
    {
        "id": "choice_k5_with_descriptions",
        "state": "Please route this support request appropriately.",
        "question": {
            "type": "choice",
            "instructions": "Which department fits best?",
            "criteria": {
                "billing": "handles invoices, refunds, and payment disputes",
                "technical": "handles bugs, outages, and product defects",
                "sales": "handles new purchases, upgrades, and renewals",
                "shipping": "handles delivery delays, lost packages, and tracking",
                "account": "handles login issues, password resets, and account access",
            },
        },
    },
    {
        "id": "choice_k10_categories",
        "state": "Customer inquiry about product categories.",
        "question": {
            "type": "choice",
            "instructions": "Which product category is this about?",
            "criteria": [
                "electronics", "clothing", "home", "toys", "sports",
                "books", "beauty", "automotive", "garden", "office",
            ],
        },
    },
    {
        "id": "choice_k20_regions",
        "state": "Which region is the customer based in, based on their message?",
        "question": {
            "type": "choice",
            "instructions": "Which region does the customer belong to?",
            "criteria": [
                "US", "CA", "MX", "BR", "UK", "FR", "DE", "ES", "IT", "NL",
                "SE", "PL", "RU", "CN", "JP", "KR", "IN", "AU", "ZA", "EG",
            ],
        },
    },
    {
        "id": "score_k3_urgency",
        "state": "The server has been down for six hours and customers are furious.",
        "question": {
            "type": "score",
            "instructions": "How urgent is this issue?",
            "criteria": [
                "low priority, can wait",
                "medium priority, address today",
                "high priority, immediate action needed",
            ],
        },
    },
    {
        "id": "score_k5_satisfaction",
        "state": "Thanks so much, the team resolved my issue quickly and I'm very happy!",
        "question": {
            "type": "score",
            "instructions": "Rate the customer's satisfaction.",
            "criteria": [
                "very dissatisfied", "dissatisfied", "neutral", "satisfied", "very satisfied",
            ],
        },
    },
    {
        "id": "noul_default_escalation",
        "state": "This is the third time I've contacted support about this bug.",
        "question": {
            "type": "noul",
            "instructions": "Does this case need escalation?",
            "criteria": None,
        },
    },
    {
        "id": "noul_custom_textoverride",
        "state": "I think I might cancel my subscription if this isn't fixed.",
        "question": {
            "type": "noul",
            "instructions": "Is the customer at risk of churning?",
            "criteria": {
                "false": "customer seems satisfied and likely to stay",
                "true": "customer shows clear signs of churn risk",
            },
        },
    },
    {
        "id": "ane_fits_70tok_state",
        "state": state_70_tokens,
        "question": {
            "type": "choice",
            "instructions": "Does this need a refund?",
            "criteria": ["yes", "no"],
        },
    },
    {
        "id": "ane_exceeds_90tok_state",
        "state": state_90_tokens,
        "question": {
            "type": "choice",
            "instructions": "Which team should own this ticket?",
            "criteria": ["billing", "technical", "sales"],
        },
    },
    {
        "id": "empty_state_case",
        "state": "",
        "question": {
            "type": "choice",
            "instructions": "Which team should own this ticket?",
            "criteria": ["billing", "technical", "sales"],
        },
    },
    {
        "id": "mask_literal_instructions",
        "state": "Explain what <mask> means in this context.",
        "question": {
            "type": "choice",
            "instructions": (
                "The customer used the word <mask> in their message "
                "— what does <mask> most likely refer to?"
            ),
            "criteria": ["a physical object", "an abstract concept", "a person's name"],
        },
    },
    {
        "id": "chinese_state_case",
        "state": "客户说他们的包裹丢失了，要求全额退款。",
        "question": {
            "type": "choice",
            "instructions": "Which team should own this ticket?",
            "criteria": ["billing", "technical", "shipping"],
        },
    },
    {
        "id": "chinese_options_case",
        "state": "Please categorize this request appropriately.",
        "question": {
            "type": "choice",
            "instructions": "Which department fits best?",
            "criteria": {
                "账单": "账单、发票和退款相关问题",
                "技术": "错误、故障和产品缺陷",
                "销售": "新购买、升级和续订",
                "物流": "配送延迟、包裹丢失和物流跟踪",
                "账户": "登录问题、密码重置和账户访问",
            },
        },
    },
    {
        "id": "arabic_state_case",
        "state": (
            "يقول العميل "
            "إن الطرد فُقد "
            "أثناء الشحن "
            "ويطلب استرداد "
            "المبلغ بالكامل."
        ),
        "question": {
            "type": "score",
            "instructions": "How urgent is this issue?",
            "criteria": [
                "low priority, can wait",
                "medium priority, address today",
                "high priority, immediate action needed",
            ],
        },
    },
    {
        "id": "arabic_options_case",
        "state": "Please route this inquiry to the right department.",
        "question": {
            "type": "choice",
            "instructions": "Which department fits best?",
            "criteria": [
                "الفواتير",
                "الدعم الفني",
                "المبيعات",
            ],
        },
    },
    {
        "id": "choice_k4_support",
        "state": "Customer reports intermittent Wi-Fi drops on the mobile app during video calls.",
        "question": {
            "type": "choice",
            "instructions": "What is the most likely root cause category?",
            "criteria": ["network", "app", "device", "account"],
        },
    },
    {
        "id": "score_k4_priority",
        "state": "A VIP customer's payment failed during renewal and they are locked out.",
        "question": {
            "type": "score",
            "instructions": "How should this be prioritized?",
            "criteria": ["low", "medium", "high", "critical"],
        },
    },
    {
        "id": "noul_partial_override",
        "state": "The customer asked if we offer a student discount.",
        "question": {
            "type": "noul",
            "instructions": "Does a discount apply to this request?",
            "criteria": {"true": "yes, a discount applies here"},
        },
    },
]

assert len(CASES) == 20, f"expected 20 cases, got {len(CASES)}"


def bundle_record(agent, state, question, qid):
    items, _internal = agent.prepare(state, question)
    item = items[0]
    batch = collate_items(
        [item],
        agent.tok.pad_token_id,
        pad_to_multiple=agent.pad_to_multiple,
        max_length=agent.cfg.get("max_len", 512),
        shape=agent.shape,
    )
    logits, action_logits = agent.forward(batch)
    answer = agent.predict(state, question)["answers"][qid]
    collated = {
        "input_ids": to_plain(batch["input_ids"]),
        "attention_mask": to_plain(batch["attention_mask"]),
        "marker_pos": to_plain(batch["marker_pos"]),
        "marker_mask": to_plain(batch["marker_mask"]),
        "qtype": to_plain(batch["qtype"]),
        "padded_length": int(batch["input_ids"].shape[1]),
    }
    return {
        "collated": collated,
        "logits": to_plain(logits),
        "action_logits": to_plain(action_logits),
        "answer": to_plain(answer),
    }


def summary_value(answer_dict):
    if answer_dict is None:
        return "skipped"
    kind = answer_dict["type"]
    if kind == "choice":
        return answer_dict["choice"]
    if kind == "score":
        return answer_dict["score"]
    return answer_dict["noul"]


def main():
    general_agent = Agent("models/general", compute_units="cpu_gpu", local_files_only=True)
    ane_agent = ANEAgent("models/ane", compute_units="cpu_ne")

    special_tokens = {
        "cls": general_agent.tok.cls_token_id,
        "sep": general_agent.tok.sep_token_id,
        "pad": general_agent.tok.pad_token_id,
        "mask": general_agent.tok.mask_token_id,
    }

    fixtures = {"special_tokens": special_tokens, "cases": []}
    summary_lines = []

    for case in CASES:
        case_id = case["id"]
        state = case["state"]
        question = {"q": case["question"]}

        prepared_items, _ = general_agent.prepare(state, question)
        prepared_item = prepared_items[0]
        sequence_length = len(prepared_item["ids"])
        exceeds_ane = sequence_length > ANE_MAX_LENGTH

        general_result = bundle_record(general_agent, state, question, "q")
        ane_result = None if exceeds_ane else bundle_record(ane_agent, state, question, "q")

        record = {
            "id": case_id,
            "state": state,
            "question": case["question"],
            "token_ids": prepared_item["ids"],
            "markers": prepared_item["markers"],
            "qtype": prepared_item["qtype"],
            "sequence_length": sequence_length,
            "exceeds_ane": exceeds_ane,
            "general": general_result,
            "ane": ane_result,
        }
        fixtures["cases"].append(record)

        general_answer = general_result["answer"]
        ane_answer = ane_result["answer"] if ane_result else None
        if ane_answer is not None:
            general_value, ane_value = summary_value(general_answer), summary_value(ane_answer)
            if general_answer["type"] == "choice":
                if general_value != ane_value:
                    print(f"WARNING [{case_id}]: general choice {general_value!r} != ane choice {ane_value!r}")
            else:
                if abs(float(general_value) - float(ane_value)) >= 0.01:
                    print(
                        f"WARNING [{case_id}]: general/ane value differs by "
                        f">= 0.01 ({general_value} vs {ane_value})"
                    )
            general_probs = general_answer.get("probabilities")
            ane_probs = ane_answer.get("probabilities")
            if general_probs is not None and ane_probs is not None:
                for label, general_p in general_probs.items():
                    ane_p = ane_probs.get(label)
                    if ane_p is not None and abs(general_p - ane_p) >= 0.01:
                        print(
                            f"WARNING [{case_id}]: probability for {label!r} differs by "
                            f">= 0.01 ({general_p} vs {ane_p})"
                        )

        kind = case["question"]["type"]
        if kind == "score":
            k = len(case["question"]["criteria"])
        elif kind == "choice":
            criteria = case["question"]["criteria"]
            k = len(criteria)
        else:
            k = "n/a"
        general_value = summary_value(general_answer)
        ane_value = summary_value(ane_answer) if ane_answer is not None else "skipped"
        summary_lines.append(
            f"{case_id} type={kind} k={k} seq_len={sequence_length} "
            f"general={general_value} ane={ane_value}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(fixtures, ensure_ascii=False, indent=2))

    for line in summary_lines:
        print(line)
    print(f"Wrote {len(fixtures['cases'])} cases to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
