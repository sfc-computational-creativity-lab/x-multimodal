from torch.utils.data import DataLoader
from src.datasets.media_eval_eim import MediaEvalEiMDataset
from src.models.valence_arousal_regressions.model import ValenceArousalModel
from src.trainers.valence_arousal_regressions import ValenceArousalRegressionsTrainer


dataset = MediaEvalEiMDataset(
        audio_dir='/data2/yamad/clips',
        arousal_average_path='/data2/yamad/annotations/arousal_cont_average.csv',
        valence_average_path='/data2/yamad/annotations/valence_cont_average.csv',
        )
model = ValenceArousalModel()


data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

trainer = ValenceArousalRegressionsTrainer(
        model=model,
        n_epoch=100,
        data_loader=data_loader,
        )

trainer.train()
