# DS-post-interview-assignment

---

## 目的

實作一個 Encoder ，能利用多種 Tasks 訓練，並支援 Save / Load 功能。

為了簡化，本實作中的 Data 皆為一維 (不考慮 batch_size )。

---

## 如何開始

0. 請 fork 此 repository

1. 安裝開發環境

    1. 安裝 [pipenv](https://github.com/pypa/pipenv)
    2. 使用 pipenv 安裝必要套件：
    ```
    $ pipenv --three
    ```

    ```
    $ pipenv install --dev
    ```

    3. 進入虛擬環境：
    ```
    $ pipenv shell
    ```

    4. 若需要額外安裝套件
    ```
    $ pipenv install <package_name>
    ```
    請務必確認有成功產生新的 Pipfile.lock 檔案。

2. 執行測試

    ```
    $ pytest
    ```
    此時應會發現測試不會通過。

3. 實作

    實作所有程式碼中的 TODO 部分，並使測試通過。

    注意：
    並非所有要求都有出現在測試中，有些需額外實作的部分，請詳細閱讀 TODO ，若能補上這些部分的測試更佳。
    請勿修改已存在的測試！！！

---

## 評估項目
    
我們期待能從您的作答中，了解您對以下項目的掌握度：

1. Tensorflow 的基礎操作
2. 演算法/模型設計
3. 程式碼架構設計
4. 版本控制系統使用習慣
5. 程式碼可讀性
    
