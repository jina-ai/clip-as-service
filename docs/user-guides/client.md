# Client API

CLIP-as-service is designed in a client-server architecture. You can use `clip_client` to send images and texts to the server and receive the responses from the server. Right now, `clip_client` provides encoding, ranking, indexing, and searching functionalities. Additionally, it has many nice designs for speeding up the processing of a large amount of data:

- Streaming: request sending is *not* blocked by the response receiving. Sending and receiving are two separate streams that run in parallel. Both are independent and each have separate internal buffer.
- Batching: large requests are segmented into small batches and send in a stream. 
- Low memory footprint: only load data when needed.
- Sync/async interface: provide `async` interface that can be easily integrated into other asynchronous system.
- Auto-detect images and text input.
- Support gRPC, HTTP, Websocket protocols with their TLS counterparts. 


```{tip}
You will need to install `clip_client` first in Python 3.7+: `pip install clip-client`.
```

(construct-client)=
## Construct client

To use `clip_client`, you need to first construct a Client object, e.g.:

```python
from clip_client import Client

c = Client('grpc://0.0.0.0:23456')
```

The URL-like scheme `grpc://0.0.0.0:23456` is what you get after {ref}`running the server<server-address>`. The scheme follows the format `scheme://netloc:port`:


| Field    | Description                                                                                                                                                                                 | Example       |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| `scheme` | The protocol of the server, must be one of `grpc`, `websocket`, `http`, `grpcs`, `websockets`, `https`. Protocols end with `s` are TLS encrypted. This must match with the server protocol. | `grpc`        |
| `netloc` | The server's IP address or hostname                                                                                                                                                         | `192.168.0.3` |
| `port`   | The public port of the server                                                                                                                                                               | `51234`       |
            

Jina AI provides a hosted service for CLIP models. Refer [here](hosting/by-jina#by-jina-python) for more details on how to connect to the hosted service.

## Encoding

`clip_client` provides {func}`~clip_client.client.Client.encode` function that allows you to send sentences, images to the server in a streaming and sync/async manner. Encoding here means getting the fixed-length vector representation of a text or image.

{func}`~clip_client.client.Client.encode` supports two basic input types:

- **An iterable of `str`**, e.g. `List[str]`, `Tuple[str]`, `Generator[str]` are all acceptable.
- **An iterable of {class}`~docarray.document.Document`**, e.g. `List[Document]`, {class}`~docarray.array.document.DocumentArray`, `Generator[Document]` are all acceptable.

Depending on the input, the output of {func}`~clip_client.client.Client.encode` is different:

- If the input is an iterable of `str`, then the output will be a `numpy.ndarray`.
- If the input is an iterable of `Document`, then the output will be a `DocumentArray`.

Now let's look at these two cases in details.

### Input as iterable of strings

- Input: each string element is auto-detected as a sentence or an image.
- Output: a `[N, D]` shape `numpy.ndarray`, where `N` is the length of the input and `D` is the CLIP embedding size. Each row corresponds to the embedding of the input object.

Any URI-like string, including relative, absolute file path, http/https path, data URI string will be considered as an image. Otherwise, it will be considered as a sentence. 

For example,

```python
from clip_client import Client

c = Client('grpc://0.0.0.0:23456')

r = c.encode(
    [
        'she smiled, with pain',
        'apple.png',
        'https://clip-as-service.jina.ai/_static/favicon.png',
        'data:image/gif;base64,R0lGODlhEAAQAMQAAORHHOVSKudfOulrSOp3WOyDZu6QdvCchPGolfO0o/XBs/fNwfjZ0frl3/zy7////wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAkAABAALAAAAAAQABAAAAVVICSOZGlCQAosJ6mu7fiyZeKqNKToQGDsM8hBADgUXoGAiqhSvp5QAnQKGIgUhwFUYLCVDFCrKUE1lBavAViFIDlTImbKC5Gm2hB0SlBCBMQiB0UjIQA7',
    ]
)
print(r)
```

gives you

```text
[[-0.09136295  0.42720157 -0.05784469 ... -0.42873043  0.04472527
   0.4437953 ]
 [ 0.43152636  0.1563695  -0.09363698 ... -0.11514216  0.1865044
   0.15025651]
 [ 0.42862126  0.17757078  0.08584607 ...  0.23284511 -0.00929402
   0.10993651]
 [ 0.4706376  -0.01384148  0.3877237  ...  0.1995864  -0.22621225
  -0.4837676 ]]
```

### Input as iterable of Documents

```{tip}
This feature uses [DocArray](https://docarray.jina.ai), which is installed together with `clip_client` as an upstream dependency. You do not need to install DocArray separately.
```

If auto-detection on a list of raw string is too "sci-fi" to you, then you may use `docarray.Document` to make the input more explicit and organized. `Document` can be used as a container to easily represent a sentence or an image.

- Input: each `Document` must be filled with `.text` or `.uri` or `.blob` or `.tensor` attribute. 
  - `Document` filled with `.text` is considered as sentence;
  - `Document` filled with `.uri` or `.blob` or `.tensor` is considered as image. If `.tensor` is filled, then its shape must be in `[H, W, C]` format.
- Output: a `DocumentArray` of the same input length. Each `Document` object in it is the same one from the input and is now filled with `.embedding` attribute. The order of the output is the same as the input.

```{note}
If the input `Document` is filled with both `.text` and `.uri`, then `.text` will be used.
```

```{caution}
The correctness of result and the order of output rely on the uniqueness of id of the input `Document`. The id will be implicitly generated if not provided. If you set the id manually, then you must make sure the id is unique, otherwise the results will not be complete.
```

The explicitness comes from now you have to put the content into the `Document` attributes. For example, we can rewrite the above example as below:

```python
from clip_client import Client
from docarray import Document

c = Client('grpc://0.0.0.0:23456')

da = [
    Document(text='she smiled, with pain'),
    Document(uri='apple.png'),
    Document(uri='apple.png').load_uri_to_image_tensor(),
    Document(blob=open('apple.png', 'rb').read()),
    Document(uri='https://clip-as-service.jina.ai/_static/favicon.png'),
    Document(
        uri='data:image/gif;base64,R0lGODlhEAAQAMQAAORHHOVSKudfOulrSOp3WOyDZu6QdvCchPGolfO0o/XBs/fNwfjZ0frl3/zy7////wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAkAABAALAAAAAAQABAAAAVVICSOZGlCQAosJ6mu7fiyZeKqNKToQGDsM8hBADgUXoGAiqhSvp5QAnQKGIgUhwFUYLCVDFCrKUE1lBavAViFIDlTImbKC5Gm2hB0SlBCBMQiB0UjIQA7'
    ),
]

r = c.encode(da)
```

Instead of sending a list of `Document`, you can also wrap it with a `DocumentArray` and then send it:

```python
r = c.encode(DocumentArray(da))
```

Now that the return result is a `DocumentArray`, we can get a summary of it using `r.summary()`.

```text
╭──────────────────────────── Documents Summary ─────────────────────────────╮
│                                                                            │
│   Length                        6                                          │
│   Homogenous Documents          False                                      │
│   4 Documents have attributes   ('id', 'mime_type', 'uri', 'embedding')    │
│   1 Document has attributes     ('id', 'mime_type', 'text', 'embedding')   │
│   1 Document has attributes     ('id', 'embedding')                        │
│                                                                            │
╰────────────────────────────────────────────────────────────────────────────╯
╭────────────────────── Attributes Summary ───────────────────────╮
│                                                                 │
│   Attribute   Data type      #Unique values   Has empty value   │
│  ─────────────────────────────────────────────────────────────  │
│   embedding   ('ndarray',)   6                False             │
│   id          ('str',)       6                False             │
│   mime_type   ('str',)       5                False             │
│   text        ('str',)       2                False             │
│   uri         ('str',)       4                False             │
│                                                                 │
╰─────────────────────────────────────────────────────────────────╯
```

To get the embedding of all Documents, simply call `r.embeddings`:

```text
[[-0.09136295  0.42720157 -0.05784469 ... -0.42873043  0.04472527
   0.4437953 ]
 [ 0.43152636  0.1563695  -0.09363698 ... -0.11514216  0.1865044
   0.15025651]
 [ 0.43152636  0.1563695  -0.09363698 ... -0.11514216  0.1865044
   0.15025651]
 [ 0.42862126  0.17757078  0.08584607 ...  0.23284511 -0.00929402
   0.10993651]
 [ 0.4706376  -0.01384148  0.3877237  ...  0.1995864  -0.22621225
  -0.4837676 ]]
```

```{tip}
Reading an image file into bytes and put into `.blob` is possible as shown above. However, it is often unnecessary. Especially if you have a lot of images, loading all of them into memory is not a good idea. Rule of thumb, always use `.uri` and trust `clip_client` to handle it well. 
```

### Async encoding

To encode `Document` in an asynchronous manner, one can use {func}`~clip_client.client.Client.aencode`.

```{tip}
Despite the sexy word "async", many data scientists have misconceptions about asynchronous behavior. And their motivation of using async function is often wrong. _Async is not a silver bullet._ In a simple language, you will only need `.aencode()` when there is another concurrent task that is also async. Then you want to "overlap" the time spending of these two tasks.

If your system is sync by design, there is nothing wrong about it. Go with `encode()` until you see a clear advantage of using `aencode()`, or until your boss tell you to do so.   
```

In the following example, there is another job `another_heavylifting_job` to represent a job like writing to database, downloading large file.

```python
import asyncio
from clip_client import Client

c = Client('grpc://0.0.0.0:23456')


async def another_heavylifting_job():
    # can be writing to database, downloading large file
    # big IO ops
    await asyncio.sleep(3)


async def main():
    t1 = asyncio.create_task(another_heavylifting_job())
    t2 = asyncio.create_task(c.aencode(['hello world'] * 100))
    await asyncio.gather(t1, t2)


asyncio.run(main())
```

The final time cost will be less than `3s + time(t2)`.

(rank-api)=
## Ranking

```{tip}
This feature is only available with `clip_server>=0.3.0`.
```

One can also rank cross-modal matches via {meth}`~clip_client.client.Client.rank` or {meth}`~clip_client.client.Client.arank`. First construct a cross-modal `Document` where the root contains an image and `.matches` contain sentences to rerank. One can also construct text-to-image rerank as below:

````{tab} Given image, rank sentences

```python
from docarray import Document

d = Document(
    uri='.github/README-img/rerank.png',
    matches=[
        Document(text=f'a photo of a {p}')
        for p in (
            'control room',
            'lecture room',
            'conference room',
            'podium indoor',
            'television studio',
        )
    ],
)
```

````

````{tab} Given sentence, rank images

```python
from docarray import Document

d = Document(
    text='a photo of conference room',
    matches=[
        Document(uri='.github/README-img/4.png'),
        Document(uri='.github/README-img/9.png'),
        Document(uri='https://clip-as-service.jina.ai/_static/favicon.png'),
    ],
)
```

````



Then call `rank`, you can feed it with multiple Documents as a list:

```python
from clip_client import Client

c = Client(server='grpcs://demo-cas.jina.ai:2096')
r = c.rank([d])

print(r['@m', ['text', 'scores__clip_score__value']])
```

Finally, in the return you can observe the matches are re-ranked according to `.scores['clip_score']`:

```text
[['a photo of a television studio', 'a photo of a conference room', 'a photo of a lecture room', 'a photo of a control room', 'a photo of a podium indoor'], 
[0.9920725226402283, 0.006038925610482693, 0.0009973491542041302, 0.00078492151806131, 0.00010626466246321797]]
```

(indexing)=
## Indexing

```{tip}
This feature is only available with `clip_client>=0.7.0`, and the server is running with 
a FLOW consisting of encoder and indexer.
``` 

You can index Documents via {func}`~clip_client.client.Client.index` or {func}`~clip_client.client.Client.aindex`. 

```python
from clip_client import Client
from docarray import Document

c = Client('grpc://0.0.0.0:23456')

da = [
    Document(text='she smiled, with pain'),
    Document(uri='apple.png'),
    Document(uri='apple.png').load_uri_to_image_tensor(),
    Document(blob=open('apple.png', 'rb').read()),
    Document(uri='https://clip-as-service.jina.ai/_static/favicon.png'),
    Document(
        uri='data:image/gif;base64,R0lGODlhEAAQAMQAAORHHOVSKudfOulrSOp3WOyDZu6QdvCchPGolfO0o/XBs/fNwfjZ0frl3/zy7////wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAkAABAALAAAAAAQABAAAAVVICSOZGlCQAosJ6mu7fiyZeKqNKToQGDsM8hBADgUXoGAiqhSvp5QAnQKGIgUhwFUYLCVDFCrKUE1lBavAViFIDlTImbKC5Gm2hB0SlBCBMQiB0UjIQA7'
    ),
]

r = c.index(da)
```
Now that the return result is a DocumentArray, we can get a summary of it.

```text
╭──────────────────────────── Documents Summary ─────────────────────────────╮
│                                                                            │
│   Length                        6                                          │
│   Homogenous Documents          False                                      │
│   4 Documents have attributes   ('id', 'mime_type', 'uri', 'embedding')    │
│   1 Document has attributes     ('id', 'mime_type', 'text', 'embedding')   │
│   1 Document has attributes     ('id', 'embedding')                        │
│                                                                            │
╰────────────────────────────────────────────────────────────────────────────╯
╭────────────────────── Attributes Summary ───────────────────────╮
│                                                                 │
│   Attribute   Data type      #Unique values   Has empty value   │
│  ─────────────────────────────────────────────────────────────  │
│   embedding   ('ndarray',)   6                False             │
│   id          ('str',)       6                False             │
│   mime_type   ('str',)       5                False             │
│   text        ('str',)       2                False             │
│   uri         ('str',)       4                False             │
│                                                                 │
╰─────────────────────────────────────────────────────────────────╯
```

The `embedding` is the output of the encoder, which is a 512-dim vector. 
Now we can use the indexer to search for the indexed Documents.


(searching)=
## Searching

```{tip}
This feature is only available with `clip_client>=0.7.0`, and the server is running with 
a FLOW consisting of encoder and indexer.
``` 

You can use {func}`~clip_client.client.Client.search` or {func}`~clip_client.client.Client.asearch`
to search for relevant Documents in the index for a given query.

```python
from clip_client import Client

c = Client('grpc://0.0.0.0:23456')

result = c.search(['smile'], limit=2)

print(result['@m', ['text', 'scores__cosine']])
```

The results will look like this, the most relevant doc is "she smiled, with pain" with the cosine distance of 0.096. And the apple image has the cosine distance of 0.799.
```text
[['she smiled, with pain', ''], [{'value': 0.09604918956756592}, {'value': 0.7994111776351929}]]
```
You can set the `limit` parameter (default is `10`) to control the number of the most similar documents to be retrieved.



(profiling)=
## Profiling

You can use {func}`~clip_client.client.Client.profile` to give a quick test on the server to make sure everything is good. 

```python
from clip_client import Client

c = Client('grpc://0.0.0.0:23456')

c.profile()
```

This give you a tree-like table showing the latency and percentage.

```text
 Roundtrip  16ms  100%                                                          
├──  Client-server network  12ms  75%                                           
└──  Server  4ms  25%                                                           
    ├──  Gateway-CLIP network  0ms  0%                                          
    └──  CLIP model  4ms  100%      
```

Under the hood, `.profile()` sends a single empty Document to the CLIP-server for encoding and calculates a summary of latency. The above tree can be read as follows:  

- From calling `client.encode()` to returning the results, everything counted, takes 16ms to finish.
- Among them the time spent on the server is 4ms, the remaining 12ms is spent on the client-server communication, request packing, response unpacking.
- During the 4ms server processing time, CLIP model takes 4ms, whereas the [Gateway](https://docs.jina.ai/fundamentals/architecture-overview/#architecture-overview) to CLIP communication takes no time.

`.profile()` can also take a string argument and asks CLIP-server to encode it. This string can be a sentence, local/remote image file URI. For example:

```python
c.profile('hello, world')
c.profile('apple.png')
c.profile('https://docarray.jina.ai/_static/favicon.png')
```


Single query latency is often very fluctuated. Running `.profile()` multiple times may give you different results. Nonetheless, it helps you understand who to blame if CLIP-as-service is running slow for you: the network? the computation? But certainly not this software itself.


## Best practices

In this section, we will show you some best practices for using this client. We will use encoding as an example. The same applies to all other methods.

### Control batch size

You can specify `.encode(..., batch_size=8)` to control how many `Document`s are sent in each request. You can play this number and find the perfect balance between network transmission and GPU utilization. 

Intuitively, setting `batch_size=1024` should result in very high GPU utilization on each request. However, a large batch size like this also means sending each request would take longer. Given that `clip-client` is designed with request and response streaming, large batch size would not benefit from the time overlapping between request streaming and response streaming.


### Show progressbar

You can use `.encode(..., show_progress=True)` to turn on the progress bar.

```{figure} images/client-pgbar.gif
:width: 80%
```

```{hint}
Progress bar may not show up in the PyCharm debug terminal. This is an upstream issue of `rich` package.
```

### Processing large number of Documents

Here are some suggestions when encoding a large number of `Document`s:

1. Use `Generator` as input to load data on-demand. You can put your data into a Generator and feed to `.encode`:
    ```python
    def data_gen():
        for _ in range(100_000):
            yield Document(uri=...)


    c = Client(...)
    c.encode(data_gen())
    ```
    Yield raw strings is also acceptable, e.g. to encode all images under a directory, you can simply do:
    ```python
    from glob import iglob

    c.encode(iglob('**/*.png'))
    ```
2. Adjust `batch_size`.
3. Turn on the progressbar.

````{danger}
In any case, avoiding the following coding:

```python
for d in big_list:
    c.encode([d])
```

This is extremely slow as only one document is encoded at a time, it is a bad utilization of the network and not leveraging any duplex streaming.
````

### Custom callback

`clip_client` by default collects all the results and returns them to users. However, if you want to process the results on-the-fly, you can also pass a callback function when sending the request. For example, you can use the callback to save the results to a database, or render the results to a webpage. Specifically, you can specify any of the three callback functions: `on_done`, `on_error`, and `on_always`.

- `on_done` is executed while streaming, after successful completion of each request
- `on_error` is executed while streaming, whenever an error occurs in each request
- `on_always` is always performed while streaming, no matter the success or failure of each request

Note that these callbacks only work for requests (and failures) inside the stream. For `on_error`, if the failure is due to an error happening outside of streaming, then it will not be triggered. For example, a `SIGKILL` from the client OS during the handling of the request, or a networking issue, will not trigger the callback. Learn more about [handling exceptions in `on_error`](https://docs.jina.ai/fundamentals/client/client/#handle-exceptions-in-callbacks).

Callback functions in expect a `Response` of the type DataRequest, which contains resulting Documents, parameters, and other information. Learn more about [handling `DataRequest` in callbacks](https://docs.jina.ai/fundamentals/client/client/#handle-datarequest-in-callbacks).


### Client parallelism

In case you instanciate a `clip_client` object using the `grpc` protocol, keep in mind that `grpc` clients cannot be used in a multi-threaded environment (check [this gRPC issue](https://github.com/grpc/grpc/issues/25364) for reference).
What you should do, is to rely on asynchronous programming or multi-processing rather than multi-threading.

To use `clip_client` in a Flask application, you can introduce multi-processing based parallelism to your app using `gunicorn`: 

```bash
gunicorn -w 4 -b 127.0.0.1:4000 myproject:app
```

To use `clip_client` in a FastAPI application, you have to manually restrict the thread number to 1 at the starting state of the app:

```python   
import uvicorn
from fastapi import FastAPI
from clip_client import Client
from anyio.lowlevel import RunVar
from anyio import CapacityLimiter

c = Client('grpc://0.0.0.0:51001')
app = FastAPI()

@app.on_event("startup")
def startup():
    print("start")
    RunVar("_default_thread_limiter").set(CapacityLimiter(1))

@app.post("/")
def encode():
    r =  c.encode(['Hello world', 'Hello Jina'])
    print(r)
```

Then it can run with multiprocessing using

```bash
gunicorn myproject:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:4000
```

## Appendix: Plain HTTP request via `curl`

```{tip}
Sending large embeddings over plain HTTP is often not the best idea. Websocket is often a better choice, allows one to call clip-server from Javascript with much better performance.   
```

If your {ref}`server is spawned<flow-config>` with `protocol: http` and `cors: True`, then you do not need to call the server via Python client. You can simply do it via `curl` or Javascript by sending a JSON to `http://address:port/post`. Notice, the `/post` endpoint at the end. For example,

To encode sentences:

```{code-block} bash
---
emphasize-lines: 3
---
curl -X POST http://demo-cas.jina.ai:51000/post \ 
     -H 'Content-Type: application/json' \
     -d '{"data":[{"text": "First do it"}, {"text": "then do it right"}, {"text": "then do it better"}], "execEndpoint":"/"}'
```

To encode a local image, you need to load it as base64 string and put into the `blob` field, and be careful with the quotes there:

```{code-block} bash
---
emphasize-lines: 3
---
curl -X POST http://demo-cas.jina.ai:51000/post \ 
     -H 'Content-Type: application/json' \
     -d '{"data":[{"text": "First do it"}, {"blob":"'"$( base64 test-1.jpeg)"'" }], "execEndpoint":"/"}'
```

To encode a remote image, you can simply put its address into `uri` field:

```{code-block} bash
---
emphasize-lines: 3
---
curl -X POST http://demo-cas.jina.ai:51000/post \ 
     -H 'Content-Type: application/json' \
     -d '{"data":[{"text": "First do it"}, {"uri": "https://clip-as-service.jina.ai/_static/favicon.png"}], "execEndpoint":"/"}'
```

Run it, you will get:

```json
{"header":{"requestId":"8b1f4b419bc54e95ab4b63cc086233c9","status":null,"execEndpoint":"/","targetExecutor":""},"parameters":null,"routes":[{"executor":"gateway","startTime":"2022-04-01T15:24:28.267003+00:00","endTime":"2022-04-01T15:24:28.328868+00:00","status":null},{"executor":"clip_t","startTime":"2022-04-01T15:24:28.267189+00:00","endTime":"2022-04-01T15:24:28.328748+00:00","status":null}],"data":[{"id":"b15331b8281ffde1e9fb64005af28ffd","parent_id":null,"granularity":null,"adjacency":null,"blob":null,"tensor":null,"mime_type":"text/plain","text":"hello, world!","weight":null,"uri":null,"tags":null,"offset":null,"location":null,"embedding":[-0.022064208984375,0.1044921875, ..., -0.1363525390625,-0.447509765625],"modality":null,"evaluations":null,"scores":null,"chunks":null,"matches":null}]}
```

The embedding is inside `.data[].embedding`. If you have [jq](https://stedolan.github.io/jq/) installed, you can easily filter the embeddings out via:

```{code-block} bash
---
emphasize-lines: 4
---
curl -X POST https://demo-cas.jina.ai:8443/post \
     -H 'Content-Type: application/json' \
     -d '{"data":[{"text": "hello, world!"}, {"blob":"'"$( base64 test-1.jpeg)"'" }], "execEndpoint":"/"}' | \
     jq -c '.data[] | .embedding'
```

```text
[-0.022064208984375,0.1044921875,...]
[-0.0750732421875,-0.166015625,...]
```

To connect to the CLIP server hosted by Jina AI, please refer to [this page](/hosting/by-jina#by-jina-curl).