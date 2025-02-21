import cv2
import pytesseract
import csv
import os
import re
from seg import U2NETP
from GeoTr import GeoTr
import torch
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import warnings

pytesseract.pytesseract.tesseract_cmd = Path('Tesseract-OCR') / 'tesseract.exe'
os.environ['TESSDATA_PREFIX'] = str(Path('Tesseract-OCR') / 'tessdata')
warnings.filterwarnings('ignore')

class GeoTr_Seg(nn.Module):
    def __init__(self):
        super(GeoTr_Seg, self).__init__()
        self.msk = U2NETP(3, 1)
        self.GeoTr = GeoTr(num_attn_layers=6)

    def forward(self, x):
        msk, _1, _2, _3, _4, _5, _6 = self.msk(x)
        msk = (msk > 0.5).float()
        x = msk * x
        bm = self.GeoTr(x)
        bm = (2 * (bm / 286.8) - 1) * 0.99
        return bm

def reload_model(model, path=""):
    if path:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location='cpu')
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

def reload_segmodel(model, path=""):
    if path:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location='cpu')
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if k[6:] in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

def process_and_save_image(input_path: str, output_path: str) -> Image:
    """Обработка нейронкой и сохранение изображения"""
    GeoTr_Seg_model = GeoTr_Seg()
    reload_segmodel(GeoTr_Seg_model.msk, str(Path("model_pretrained") / "geotr.pth"))
    reload_model(GeoTr_Seg_model.GeoTr, str(Path("model_pretrained") / "geotr.pth"))
    GeoTr_Seg_model.eval()

    # Загрузка и предобработка изображения
    input_image = Image.open(input_path).convert('RGB')
    im_ori = np.array(input_image)[:, :, :3] / 255.
    h, w, _ = im_ori.shape
    im = cv2.resize(im_ori, (288, 288))
    im = im.transpose(2, 0, 1)
    im = torch.from_numpy(im).float().unsqueeze(0)

    # Обработка моделью
    with torch.no_grad():
        bm = GeoTr_Seg_model(im)
        bm = bm.cpu()
        bm0 = cv2.resize(bm[0, 0].numpy(), (w, h))
        bm1 = cv2.resize(bm[0, 1].numpy(), (w, h))
        bm0 = cv2.blur(bm0, (3, 3))
        bm1 = cv2.blur(bm1, (3, 3))
        lbl = torch.from_numpy(np.stack([bm0, bm1], axis=2)).unsqueeze(0)

        out = F.grid_sample(torch.from_numpy(im_ori).permute(2, 0, 1).unsqueeze(0).float(), lbl, align_corners=True)
        img_geo = ((out[0] * 255).permute(1, 2, 0).numpy()).astype(np.uint8)

    # Сохранение результата
    result_image = Image.fromarray(img_geo)
    result_image.save(output_path)
    print(f'Изображение успешно сохранено в {output_path}')
    return result_image


def clean_text(text: str) -> str:
    """Очистка текста, считанного с обработанного изображения"""
    return re.sub(r'["\'\\;/\[\]{}()%$^&#@|©`<’‘_№]', '', text)


def cv2_rebuild(image_path: str, output_dir: Path) -> Image:
    """Предобработка изображения и передача его в нейронку"""

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(image_path).name
    output_path = str(output_dir / f"{filename}")

    image = cv2.imread(image_path)
    image = cv2.resize(image, (2480, 3508))

    if image is None:
        raise ValueError(f"Не удалось загрузить: {image_path}")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.bilateralFilter(gray, d=10, sigmaColor=10, sigmaSpace=75)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    cv2.imwrite(output_path, binary)
    return process_and_save_image(output_path, output_path)


def process_images(input_pattern: Path, output_dir: Path, output_csv: str,  with_txt: bool) -> None:
    """Обработка входных изображений"""
    image_paths = (
            list(input_pattern.glob("file_*.jpg")) +
            list(input_pattern.glob("file_*.jpeg")) +
            list(input_pattern.glob("file_*.png"))
    )
    files_amount = len(image_paths)
    
    if not image_paths:
        raise ValueError("Файлы не найдены по указанному пути")

    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / f'{output_csv}', 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for idx, image_path in enumerate(image_paths, 1):
            try:
                processed_image = cv2_rebuild(image_path=str(image_path), output_dir=Path('processed_images'))
                text = clean_text(pytesseract.image_to_string(processed_image, lang='rus'))

                if with_txt:
                    with open(output_dir / f'{Path(image_path).stem}.txt', 'w', encoding='utf-8') as txt_file:
                        txt_file.write(text)

                writer.writerow([f'{Path(image_path).name}', f'{text}'])
                print(f'Обработано: {idx}/{files_amount}')
                
            except Exception as e:
                error_msg = f'Ошибка в {Path(image_path).name}: {str(e)}'
                print(error_msg)
                

if __name__ == '__main__':
    process_images(input_pattern=Path('source_images'), output_dir=Path('output_files'), output_csv='text_results.csv',
                   with_txt=True)