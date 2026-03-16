import os
import re
import json
import torch
import numpy as np


VALID_PROMPT_MODES = (
    "base_native",
    "instruct_flat",
    "instruct_templated",
    "instruct_single_turn",
)


def parse_trip_response(response):
    """Parse a trip planning response into a list of (city, stay_days) tuples.

    Ported from the official Dream eval:
    https://github.com/HKUNLP/Dream/blob/main/eval/trip_metric.py
    """
    pattern_visit = r'\d+-\d+'
    pattern_flight = r'.*Day (\d+).*from (\w+) to (\w+)'
    pattern_days = r'European cities for (\d+) days'

    days, flights, flight_days = [], [], []
    total_days = None
    for piece in response.split('\n'):
        days_match = re.findall(pattern_days, piece)
        if days_match:
            total_days = int(days_match[0])

        visit_match = re.findall(pattern_visit, piece)
        if visit_match:
            days.append(visit_match[0])
            end_day = int(visit_match[0].split('-')[1])
            if end_day == total_days:
                break
        flight_match = re.findall(pattern_flight, piece)
        if flight_match:
            flights.append(flight_match[0])

    visit_cities, parsed_plan = [], []
    for flight_day, begin_city, end_city in flights:
        flight_days.append(int(flight_day))
        if not visit_cities:
            visit_cities.append(begin_city)
        visit_cities.append(end_city)

    if not days or not flights or not visit_cities:
        return []

    last_day = int(days[-1].split('-')[1])
    flight_days = [1] + flight_days + [last_day]
    for i, visit_city in enumerate(visit_cities):
        city_stay = flight_days[i + 1] - flight_days[i] + 1
        parsed_plan.append((visit_city, city_stay))

    return parsed_plan


def trip_score_single(cities_str, durations_str, response):
    """Score a single trip planning response (exact match).

    Args:
        cities_str:    "city1**city2**city3"
        durations_str: "5**5**6"
        response:      raw model output

    Returns True if the plan exactly matches ground truth.
    """
    cleaned = response.split('<|endoftext|>')[0].split('\nTASK')[0]
    parsed_plan = parse_trip_response(cleaned)

    stays = [x for x in cities_str.split('**') if x]
    days = [int(x) for x in durations_str.split('**') if x]

    num_stays = min(len(stays), len(parsed_plan))
    if num_stays == 0:
        return False

    num_match = 0
    for i in range(num_stays):
        if stays[i] == parsed_plan[i][0] and days[i] == parsed_plan[i][1]:
            num_match += 1
        else:
            break

    return num_match == len(stays)


class TripPlanningDataset(torch.utils.data.Dataset):
    """Trip Planning dataset matching the official Dream evaluation.

    Uses the exact same prompt format and scoring as
    https://github.com/HKUNLP/Dream/blob/main/eval/eval_planning.py

    Default is 2-shot (Dream paper Table 1), derived from the 5-shot
    prompts by keeping the first 2 example tasks.

    Source: eval/data/trip_planning.json in the Dream repo.
    1600 problems total (200 per num_cities from 3 to 10).

    Prompt modes
    ------------
      base_native        — flat prompt + "\\nSOLUTION: " suffix
      instruct_flat      — same flat string (no chat template)
      instruct_templated — single user message wrapped in chat template
      instruct_single_turn — alias for the same single-turn wrapping
                             (trip planning uses a domain-specific TASK format
                             that doesn't decompose into user/assistant turns,
                             so both chat modes keep the flat content intact)
    """

    def __init__(
        self,
        tokenizer,
        num_examples=2,
        subsample=-1,
        is_base_model=False,
        prompt_mode=None,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.num_examples = num_examples

        if prompt_mode is not None:
            if prompt_mode not in VALID_PROMPT_MODES:
                raise ValueError(
                    f"Invalid prompt_mode={prompt_mode!r}. "
                    f"Must be one of {VALID_PROMPT_MODES}"
                )
            self.prompt_mode = prompt_mode
        else:
            self.prompt_mode = "base_native" if is_base_model else "instruct_templated"

        self.is_base_model = (self.prompt_mode == "base_native")

        self.tokenizer.padding_side = "left"

        self._load_data()

        _subsample_rng = np.random.RandomState(42)
        n_test = len(self.test_data)
        if 0 < subsample < n_test:
            self.subsample = _subsample_rng.choice(n_test, subsample, replace=False)
        else:
            self.subsample = np.arange(n_test)

        print(f"evaluating {len(self.subsample)} examples (trip planning)")
        print(f"Prompt mode: {self.prompt_mode}")

    def _load_data(self):
        cur_path = os.path.dirname(os.path.abspath(__file__))
        data_file = os.path.join(cur_path, "trip_planning.json")
        with open(data_file, "r") as f:
            raw = json.load(f)

        self.test_data = []
        for key in sorted(raw.keys()):
            item = raw[key]
            prompt_5shot = item["prompt_5shot"]
            splits = prompt_5shot.split("TASK:")

            if self.num_examples >= 5:
                prompt = prompt_5shot
            elif self.num_examples > 0:
                n = self.num_examples
                prompt = "TASK:".join(splits[: n + 1] + [splits[-1]])
            else:
                prompt = item.get("prompt_0shot", "TASK:" + splits[-1])

            self.test_data.append({
                "prompt": prompt,
                "cities": item["cities"],
                "durations": item["durations"],
                "num_cities": item["num_cities"],
            })

        print(f"Trip Planning: {len(self.test_data)} problems loaded")

    # ------------------------------------------------------------------
    # 3-way prompt dispatch
    # ------------------------------------------------------------------

    def create_flat_prompt(self, prompt_text):
        """Flat prompt with exactly one terminal SOLUTION cue."""
        prompt_text = prompt_text.rstrip()
        if prompt_text.endswith("SOLUTION:"):
            return prompt_text + " "
        return prompt_text + "\nSOLUTION: "

    def create_templated_instruct_prompt(self, prompt_text):
        """Wrap the flat prompt in a single user message with chat template.

        Trip planning uses a domain-specific TASK: format that doesn't
        decompose cleanly into user/assistant turns, so we keep the full
        flat prompt as a single user message.

        Preserve the terminal "SOLUTION:" cue from the flat prompt so the
        instruct model continues the same completion format shown in the
        few-shot demonstrations instead of switching into generic chatty prose.
        """
        flat_prompt = self.create_flat_prompt(prompt_text).rstrip()
        messages = [{"role": "user", "content": flat_prompt}]
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

    def create_single_turn_instruct_prompt(self, prompt_text):
        """Trip planning already uses a single user turn, so reuse that path."""
        return self.create_templated_instruct_prompt(prompt_text)

    def create_prompt(self, prompt_text):
        """Dispatch to the appropriate prompt builder based on prompt_mode."""
        if self.prompt_mode in ("base_native", "instruct_flat"):
            return self.create_flat_prompt(prompt_text)
        if self.prompt_mode == "instruct_single_turn":
            return self.create_single_turn_instruct_prompt(prompt_text)
        return self.create_templated_instruct_prompt(prompt_text)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.subsample)

    def __getitem__(self, idx):
        item = self.test_data[self.subsample[idx].item()]
        prompt = self.create_prompt(item["prompt"])
        gt = f"{item['cities']}||{item['durations']}"
        return prompt, gt, gt

    def collate_fn(self, batch):
        prompts = [item[0] for item in batch]
        questions = [item[1] for item in batch]
        answers = [item[2] for item in batch]
        encoded = self.tokenizer(
            prompts, return_tensors="pt", padding="longest"
        )
        return {
            "input_ids": encoded.input_ids,
            "attention_mask": encoded.attention_mask,
            "questions": questions,
            "answers": answers,
            "prompts": prompts,
        }
