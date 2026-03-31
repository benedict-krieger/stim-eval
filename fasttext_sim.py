import torch
import fasttext
import fasttext.util

cos = torch.nn.CosineSimilarity(dim=0)

fasttext.util.download_model('de', if_exists='ignore')  # 
ft = fasttext.load_model('cc.de.300.bin')

if __name__=='__main__':
        v = 'verließ'
        b = 'betrat'
        v_vec = torch.from_numpy(ft.get_word_vector(v))
        b_vec = torch.from_numpy(ft.get_word_vector(b))
        sim = cos(v_vec,b_vec).item()
        print(sim)