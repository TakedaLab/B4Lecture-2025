import random

PARTICIPANTS = [
    "teruya",
    "depontes",
    "ishikawa",
    "ilias",
    "futagami",
    "nakaseko",
]

SUBJECTS = [
    "解析・線形　　",
    "確率・統計　　",
    "プログラミング",
]

if __name__ == "__main__":
    start_year = int(input("1つ目の過去問の年度を入力してください: "))
    random.shuffle(PARTICIPANTS)
    print(f"{start_year}年")
    for i in range(0, 3):
        print(f"{SUBJECTS[i]}: {PARTICIPANTS[i]}")
    print(f"{start_year + 1}年")
    for i in range(3, 6):
        print(f"{SUBJECTS[i - 3]}: {PARTICIPANTS[i]}")
