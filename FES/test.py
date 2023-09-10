from model_attn import FES
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from utils_attn import get_data, load_checkpoint


if __name__ == "__main__":
    config = {
        'batch_size': 200,
        'lr': 1e-6,
    }

    epochs = 300
    name = "FES_50K_attn"

    model = FES(config)

    trainer = Trainer(gpus=-1, fast_dev_run=True)
    trainer.fit(model)

    _, _, test = get_data(batch_size=config['batch_size'])
    trainer.test(test_dataloaders=test, ckpt_path='./ckpt/epoch=069_data_10K_attn.ckpt')
