"""Connect the chapter outputs; a teaching prototype, not a complete product."""
import argparse
import importlib.util
from pathlib import Path
import json
import torch
from PIL import Image
from _common import OUT, np
from _torch_models import DigitsCNN, TinySequenceModel, seed

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True, help='black background, white single digit')
    parser.add_argument('--feedback', required=True)
    parser.add_argument('--digits', nargs='+', type=int, default=[2, 4, 7, 9, 1])
    args = parser.parse_args()
    if not args.feedback.strip():
        parser.error('feedback must not be empty')
    if len(args.digits) > 18 or any(d < 0 or d > 9 for d in args.digits):
        parser.error('provide 1..18 digits in the range 0..9')
    seed()
    cnn = DigitsCNN()
    cnn.load_state_dict(torch.load(OUT / '11_best.pt', weights_only=True))
    cnn.eval()
    pixels = np.asarray(Image.open(args.image).convert('L').resize((8, 8)), dtype=np.float32) / 255
    text_path = Path(__file__).with_name('12_text.py')
    spec = importlib.util.spec_from_file_location('learning_text', text_path)
    text_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(text_module)
    saved_text = torch.load(OUT / '12_text.pt', weights_only=True)
    vocab = saved_text['vocab']
    text_model = text_module.TextRNN(len(vocab))
    text_model.load_state_dict(saved_text['model'])
    text_model.eval()
    ids = torch.tensor([[vocab.get(char, 1) for char in args.feedback]], dtype=torch.long)
    lengths = torch.tensor([ids.shape[1]])
    saved_transformer = torch.load(OUT / '13_transformer.pt', weights_only=True)
    transformer = TinySequenceModel(**saved_transformer['config'])
    transformer.load_state_dict(saved_transformer['model'])
    transformer.eval()
    generated = torch.tensor([[d + 1 for d in args.digits] + [11]])
    with torch.no_grad():
        digit = int(cnn(torch.tensor(pixels)[None, None]).argmax(1))
        feedback_label = int(text_model(ids, lengths).argmax(1))
        for _ in range(len(args.digits) + 1):
            next_id = transformer(generated)[:, -1].argmax(1, keepdim=True)
            generated = torch.cat([generated, next_id], 1)
        output = generated[0, len(args.digits) + 1:].tolist()
    expected = [d + 1 for d in reversed(args.digits)] + [12]
    result = {
        'image_predicted_digit': digit,
        'feedback': args.feedback,
        'feedback_predicted_label': feedback_label,
        'feedback_label_meaning': '自述已理解' if feedback_label == 1 else '建议复习',
        'unknown_character_count': sum(char not in vocab for char in args.feedback),
        'input_digits': args.digits,
        'generated_token_ids': output,
        'generated_digits': [token - 1 for token in output if 1 <= token <= 10],
        'reversal_exact_including_eos': output == expected,
        'limitations': 'Three independent teaching models; the feedback model has only 12 training examples. Reversal length changes may fail. No image-text joint learning.',
    }
    path = OUT / 'learning_card_demo.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
