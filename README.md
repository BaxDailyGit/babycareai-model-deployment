# 모찌케어-model-deployment

# 문서화중입니다.

# ---legacy---

초보 부모들은 아기가 병원에 가봐야 하는 상황인지조차 파악하기 어려울 때가 많다. 특히 영유아기에 자주 발생하는 크고 작은 피부 질환들은 때로는 심각한 피부 질환과 구분하기 어려울 수 있다. '모찌케어'는 아기의 피부 병변 이미지와 추가적인 증상들을 종합적으로 분석하여 피부 질환을 분석하고 가정 내 처치 방법, 증상의 중증도, 병원 내원 필요 여부 등의 정보를 제공한다.

https://baby-care-ai-app.vercel.app

https://github.com/user-attachments/assets/09b2692a-c152-4230-9bdc-8dd68905ebef

(현재는 피부질환 예측 기능같은 경우 Amazon SageMaker 실시간 추론에서 서버리스추론으로 전환하였습니다.)

# 기능 
> ### 피부질환분석 (MVP)
> - 이미지 업로드 
> - 피부질환 예측 
> - 추가 정보 입력 (발열 여부, 가려움 등) 
> - 최종 진단 및 가이드 생성 

<br>

# 팀원 구성
<table style="width: 100%;">
<tr>
    <td align="center" style="width: 49%;"><img src="https://github.com/user-attachments/assets/31670ccd-0bdb-4697-bfc4-7962e7e01e69" width="130px;" alt=""></a></td>
    <td align="center" style="width: 49%;"><img src="https://avatars.githubusercontent.com/u/65113282?v=4" width="130px;" alt=""></a></td>
    <td align="center" style="width: 49%;"><img src="https://avatars.githubusercontent.com/u/108132550?v=4" width="130px;" alt=""></a></td>
    <td align="center" style="width: 49%;"><img src="https://avatars.githubusercontent.com/u/99312529?v=4" width="130px;" alt=""></a></td>
    <td align="center" style="width: 49%;"><img src="https://avatars.githubusercontent.com/u/59814042?v=4" width="130px;" alt=""></a></td>
</tr>
<tr>
    <td align="center"><b>김민지</b></a></td>
    <td align="center"><a href="https://github.com/eundoobidoobab"><b>조은수</b></a></td>
    <td align="center"><a href="https://github.com/hanjh193"><b>한재현</b></a></td>
    <td align="center"><a href="https://github.com/BaxDailyGit"><b>백승진</b></a></td>
    <td align="center"><a href="https://github.com/gustn1029"><b>김현수</b></a></td>
</tr>
<tr>
    <td align="center">기획자</td>
    <td align="center">기획자</td>
    <td align="center">AI 개발자</td>
    <td align="center">백엔드 개발자</td>
    <td align="center">프론트엔드 개발자</td>
</tr>
</table>

# 시스템 아키텍처
![image](https://github.com/user-attachments/assets/326d5366-600b-4e24-a440-593697053262)


# 예측 모델 배포 과정

> ### 사전 준비사항
> - SageMaker Notebook 인스턴스 환경 준비
> - IAM Role, S3 접근 권한 설정

> [!NOTE]
> 이제 AI 개발자께서 피부질환 예측 모델을 학습시켜 최신 pth 포맷 파일을 전달해 주실 때마다 다음 과정을 반복합니다.
> 
> ### 1. 모델 아티팩트 압축 후 S3 업로드
> ### 2. PyTorchModel 객체 생성
>    - 이때 필요시 inference.py 및 requirements.txt 수정
> ### 4. 모델 배포(엔드포인트 생성)
> ### 5. 백엔드 서버에서 달라진 엔드포인트 수정

