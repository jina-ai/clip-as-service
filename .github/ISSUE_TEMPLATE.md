**Prerequisites**

> Please fill in by replacing `[ ]` with `[x]`.

* [ ] Are you running the latest `bert-as-service`?
* [ ] Did you follow [the installation](https://github.com/hanxiao/bert-as-service#install) and [the usage](https://github.com/hanxiao/bert-as-service#usage) instructions in `README.md`?
* [ ] Did you check the [FAQ list in `README.md`](https://github.com/hanxiao/bert-as-service#speech_balloon-faq)?
* [ ] Did you perform [a cursory search on existing issues](https://github.com/hanxiao/bert-as-service/issues)?

**System information**

> Some of this information can be collected via [this script](https://github.com/tensorflow/tensorflow/tree/master/tools/tf_env_collect.sh).

- OS Platform and Distribution (e.g., Linux Ubuntu 16.04):
- TensorFlow installed from (source or binary):
- TensorFlow version:
- Python version:
- `bert-as-service` version: 
- GPU model and memory:
- CPU model and memory:

---

### Description

> Please replace `YOUR_SERVER_ARGS` and `YOUR_CLIENT_ARGS` accordingly. You can also write your own description for reproducing the issue.

I'm using this command to start the server:

```bash
bert-serving-start YOUR_SERVER_ARGS
```

and calling the server via:
```python
bc = BertClient(YOUR_CLIENT_ARGS)
bc.encode()
```

Then this issue shows up:

...