from StyleGAN import GAN
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from utils import Viusal, load_checkpoint
from pytorch_lightning.loggers import TensorBoardLogger


if __name__ == "__main__":
    batch_size = 64
    D_lr = 1e-4
    G_lr = 0.75e-4
    epochs = 250
    name = "edge"

    logger = TensorBoardLogger(f'logs', name=f'{name}')

    checkpoint_callback = ModelCheckpoint(
        filepath='./ckpt/{epoch:03d}' + f'_{name}',
        period=5,

    )
    model = GAN(batch_size, D_lr, G_lr, state='solid', debug=False, xrd=False, d2=False)

    # model.D.FES = load_checkpoint(model.D.FES, '../PES_attn/ckpt/l1_0.01_50K.ckpt')
    # model.D.FES = load_checkpoint(model.D.FES, '../PES_attn/ckpt/epoch=299_FES_50K_withT_xz.ckpt')
    model.D.FES = load_checkpoint(model.D.FES, '../PES_attn/ckpt/epoch=019_resume_FES_50K_3d.ckpt')
    for param in model.D.FES.parameters():
        param.requires_grad = False
    model.D.FES.eval()

    trainer = Trainer(gpus=-1, logger=logger, max_epochs=epochs, checkpoint_callback=checkpoint_callback,
                      callbacks=[Viusal(f'./{name}')])
    # trainer = Trainer(gpus=-1, fast_dev_run=True)
    trainer.fit(model)


