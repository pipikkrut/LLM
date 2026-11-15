CONFIG = {
    # модель (~98M параметров)
    "VOCAB_SIZE": 16000,     # тот же токенизатор, перетокенизировать данные не нужно
    "EMBEDDING_DIM": 768,
    "N_HEAD": 12,            # head_dim = 64
    "N_LAYERS": 12,
    "BLOCK_SIZE": 512,
    "DROPOUT": 0.0,

    # обучение
    "BATCH_SIZE": 16,        # микро-батч на одну GPU; больше = быстрее, пока хватает памяти
    "GRAD_ACCUM": 8,         # суммарно на все GPU: 16 * 512 * 8 = 65 536 токенов на шаг
    "MAX_STEPS": 15000,      # ~1B токенов; пересчитайте по скорости (см. лог первых 100 шагов)
    "WARMUP_STEPS": 500,
    "LEARNING_RATE": 5e-4,
    "MIN_LR_RATIO": 0.1,
    "WEIGHT_DECAY": 0.1,
    "COMPILE": False,        # True = torch.compile (быстрее, но первые шаги долгие); пробуйте после проверки скорости

    # логи и чекпоинты
    "EVAL_INTERVAL": 500,
    "EVAL_ITERS": 50,
    "LOG_INTERVAL": 50,
    "CKPT_DIR": "checkpoints",
}