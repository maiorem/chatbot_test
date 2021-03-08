import training_bot
import numpy as np
import pickle
from tensorflow.keras.models import load_model


word_to_index=pickle.load(open('word_to_index.pkl', 'rb'))
index_to_word=pickle.load(open('index_to_word.pkl', 'rb'))
encoder_model=load_model('encoder_model.h5')
decoder_model=load_model('decoder_model.h5')


# 예측을 위한 입력 생성
def make_predict_input(sentence):

    sentences = []
    sentences.append(sentence)
    sentences = training_bot.pos_tag(sentences)
    input_seq = training_bot.convert_text_to_index(sentences, word_to_index, training_bot.ENCODER_INPUT)
    
    return input_seq

# 텍스트 생성
def generate_text(input_seq):
    
    # 입력을 인코더에 넣어 마지막 상태 구함
    states = encoder_model.predict(input_seq)

    # 목표 시퀀스 초기화
    target_seq = np.zeros((1, 1))
    
    # 목표 시퀀스의 첫 번째에 <START> 태그 추가
    target_seq[0, 0] = training_bot.STA_INDEX
    
    # 인덱스 초기화
    indexs = []
    
    # 디코더 타임 스텝 반복
    while 1:
        # 디코더로 현재 타임 스텝 출력 구함
        # 처음에는 인코더 상태를, 다음부터 이전 디코더 상태로 초기화
        decoder_outputs, state_h, state_c = decoder_model.predict(
                                                [target_seq] + states)

        # 결과의 원핫인코딩 형식을 인덱스로 변환
        index = np.argmax(decoder_outputs[0, 0, :])
        indexs.append(index)
        
        # 종료 검사
        if index == training_bot.END_INDEX or len(indexs) >= training_bot.max_sequences:
            break

        # 목표 시퀀스를 바로 이전의 출력으로 설정
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = index
        
        # 디코더의 이전 상태를 다음 디코더 예측에 사용
        states = [state_h, state_c]

    # 인덱스를 문장으로 변환
    sentence = training_bot.convert_index_to_text(indexs, index_to_word)
        
    return sentence

if __name__ =="__main__":
    input_seq = make_predict_input('김치도 없네')
    sentence = generate_text(input_seq)
    print(sentence)