import mimetypes
import os
import time
import warnings
from typing import (
    overload,
    TYPE_CHECKING,
    Optional,
    Union,
    Iterator,
    Generator,
    Iterable,
    Dict,
)
from urllib.parse import urlparse
from functools import partial
from docarray import DocumentArray

if TYPE_CHECKING:
    import numpy as np
    from docarray import Document


class Client:
    def __init__(self, server: str, credential: dict = {}, **kwargs):
        """Create a Clip client object that connects to the Clip server.
        Server scheme is in the format of `scheme://netloc:port`, where
            - scheme: one of grpc, websocket, http, grpcs, websockets, https
            - netloc: the server ip address or hostname
            - port: the public port of the server
        :param server: the server URI
        :param credential: the credential for authentication {'Authentication': '<token>'}
        """
        try:
            r = urlparse(server)
            _port = r.port
            self._scheme = r.scheme
        except:
            raise ValueError(f'{server} is not a valid scheme')

        _tls = False
        if self._scheme in ('grpcs', 'https', 'wss'):
            self._scheme = self._scheme[:-1]
            _tls = True

        if self._scheme == 'ws':
            self._scheme = 'websocket'  # temp fix for the core
            if credential:
                warnings.warn(
                    'Credential is not supported for websocket, please use grpc or http'
                )

        if self._scheme in ('grpc', 'http', 'websocket'):
            _kwargs = dict(host=r.hostname, port=_port, protocol=self._scheme, tls=_tls)

            from jina import Client

            self._client = Client(**_kwargs)
            self._async_client = Client(**_kwargs, asyncio=True)
        else:
            raise ValueError(f'{server} is not a valid scheme')

        self._authorization = credential.get('Authorization', None)

    @overload
    def encode(
        self,
        content: Iterable[str],
        *,
        batch_size: Optional[int] = None,
        show_progress: bool = False,
        parameters: Optional[dict] = None,
    ) -> 'np.ndarray':
        """Encode images and texts into embeddings where the input is an iterable of raw strings.
        Each image and text must be represented as a string. The following strings are acceptable:
            - local image filepath, will be considered as an image
            - remote image http/https, will be considered as an image
            - a dataURI, will be considered as an image
            - plain text, will be considered as a sentence
        :param content: an iterator of image URIs or sentences, each element is an image or a text sentence as a string.
        :param batch_size: the number of elements in each request when sending ``content``
        :param show_progress: if set, show a progress bar
        :param parameters: the parameters for the encoding, you can specify the model to use when you have multiple models
        :return: the embedding in a numpy ndarray with shape ``[N, D]``. ``N`` is in the same length of ``content``
        """
        ...

    @overload
    def encode(
        self,
        content: Union['DocumentArray', Iterable['Document']],
        *,
        batch_size: Optional[int] = None,
        show_progress: bool = False,
        parameters: Optional[dict] = None,
    ) -> 'DocumentArray':
        """Encode images and texts into embeddings where the input is an iterable of :class:`docarray.Document`.
        :param content: an iterable of :class:`docarray.Document`, each Document must be filled with `.uri`, `.text` or `.blob`.
        :param batch_size: the number of elements in each request when sending ``content``
        :param show_progress: if set, show a progress bar
        :param parameters: the parameters for the encoding, you can specify the model to use when you have multiple models
        :return: the embedding in a numpy ndarray with shape ``[N, D]``. ``N`` is in the same length of ``content``
        """
        ...

    def encode(self, content, **kwargs):
        if isinstance(content, str):
            raise TypeError(
                f'content must be an Iterable of [str, Document], try `.encode(["{content}"])` instead'
            )

        self._prepare_streaming(
            not kwargs.get('show_progress'),
            total=len(content) if hasattr(content, '__len__') else None,
        )
        results = DocumentArray()
        with self._pbar:
            self._client.post(
                **self._get_post_payload(content, kwargs),
                on_done=partial(self._gather_result, results=results),
            )
        return self._unboxed_result(results)

    def _gather_result(self, response, results: 'DocumentArray'):
        from rich import filesize

        if not results:
            self._pbar.start_task(self._r_task)
        r = response.data.docs
        results.extend(r)
        self._pbar.update(
            self._r_task,
            advance=len(r),
            total_size=str(
                filesize.decimal(int(os.environ.get('JINA_GRPC_RECV_BYTES', '0')))
            ),
        )

    @staticmethod
    def _unboxed_result(results: 'DocumentArray'):
        if results.embeddings is None:
            raise ValueError(
                'empty embedding returned from the server. '
                'This often due to a mis-config of the server, '
                'restarting the server or changing the serving port number often solves the problem'
            )
        return (
            results.embeddings if ('__created_by_CAS__' in results[0].tags) else results
        )

    def _iter_doc(self, content) -> Generator['Document', None, None]:
        from rich import filesize
        from docarray import Document

        if hasattr(self, '_pbar'):
            self._pbar.start_task(self._s_task)

        for c in content:
            if isinstance(c, str):
                _mime = mimetypes.guess_type(c)[0]
                if _mime and _mime.startswith('image'):
                    yield Document(
                        tags={'__created_by_CAS__': True}, uri=c
                    ).load_uri_to_blob()
                else:
                    yield Document(tags={'__created_by_CAS__': True}, text=c)
            elif isinstance(c, Document):
                if c.content_type in ('text', 'blob'):
                    yield c
                elif not c.blob and c.uri:
                    c.load_uri_to_blob()
                    yield c
                elif c.tensor is not None:
                    yield c
                else:
                    raise TypeError(f'unsupported input type {c!r} {c.content_type}')
            else:
                raise TypeError(f'unsupported input type {c!r}')

            if hasattr(self, '_pbar'):
                self._pbar.update(
                    self._s_task,
                    advance=1,
                    total_size=str(
                        filesize.decimal(
                            int(os.environ.get('JINA_GRPC_SEND_BYTES', '0'))
                        )
                    ),
                )

    def _get_post_payload(self, content, kwargs):
        parameters = kwargs.get('parameters', {})
        model_name = parameters.get('model', '')
        payload = dict(
            on=f'/encode/{model_name}'.rstrip('/'),
            inputs=self._iter_doc(content),
            request_size=kwargs.get('batch_size', 8),
            total_docs=len(content) if hasattr(content, '__len__') else None,
        )
        if self._scheme == 'grpc' and self._authorization:
            payload.update(metadata=('authorization', self._authorization))
        elif self._scheme == 'http' and self._authorization:
            payload.update(headers={'Authorization': self._authorization})
        return payload

    def profile(self, content: Optional[str] = '') -> Dict[str, float]:
        """Profiling a single query's roundtrip including network and computation latency. Results is summarized in a table.
        :param content: the content to be sent for profiling. By default it sends an empty Document
            that helps you understand the network latency.
        :return: the latency report in a dict.
        """
        st = time.perf_counter()
        r = self._client.post('/', self._iter_doc([content]), return_responses=True)
        ed = (time.perf_counter() - st) * 1000
        route = r[0].routes
        gateway_time = (
            route[0].end_time.ToMilliseconds() - route[0].start_time.ToMilliseconds()
        )
        clip_time = (
            route[1].end_time.ToMilliseconds() - route[1].start_time.ToMilliseconds()
        )
        network_time = ed - gateway_time
        server_network = gateway_time - clip_time

        from rich.table import Table

        def make_table(_title, _time, _percent):
            table = Table(show_header=False, box=None)
            table.add_row(
                _title, f'[b]{_time:.0f}[/b]ms', f'[dim]{_percent * 100:.0f}%[/dim]'
            )
            return table

        from rich.tree import Tree

        t = Tree(make_table('Roundtrip', ed, 1))
        t.add(make_table('Client-server network', network_time, network_time / ed))
        t2 = t.add(make_table('Server', gateway_time, gateway_time / ed))
        t2.add(
            make_table(
                'Gateway-CLIP network', server_network, server_network / gateway_time
            )
        )
        t2.add(make_table('CLIP model', clip_time, clip_time / gateway_time))

        from rich import print

        print(t)

        return {
            'Roundtrip': ed,
            'Client-server network': network_time,
            'Server': gateway_time,
            'Gateway-CLIP network': server_network,
            'CLIP model': clip_time,
        }

    @overload
    async def aencode(
        self,
        content: Iterator[str],
        *,
        batch_size: Optional[int] = None,
        show_progress: bool = False,
    ) -> 'np.ndarray':
        ...

    @overload
    async def aencode(
        self,
        content: Union['DocumentArray', Iterable['Document']],
        *,
        batch_size: Optional[int] = None,
        show_progress: bool = False,
    ) -> 'DocumentArray':
        ...

    async def aencode(self, content, **kwargs):
        from rich import filesize

        self._prepare_streaming(
            not kwargs.get('show_progress'),
            total=len(content) if hasattr(content, '__len__') else None,
        )

        results = DocumentArray()
        async for da in self._async_client.post(
            **self._get_post_payload(content, kwargs)
        ):
            if not results:
                self._pbar.start_task(self._r_task)
            results.extend(da)
            self._pbar.update(
                self._r_task,
                advance=len(da),
                total_size=str(
                    filesize.decimal(int(os.environ.get('JINA_GRPC_RECV_BYTES', '0')))
                ),
            )

        return self._unboxed_result(results)

    def _prepare_streaming(self, disable, total):

        if total is None:
            total = 500
            warnings.warn(
                'the length of the input is unknown, the progressbar would not be accurate.'
            )

        from docarray.array.mixins.io.pbar import get_pbar

        self._pbar = get_pbar(disable)

        os.environ['JINA_GRPC_SEND_BYTES'] = '0'
        os.environ['JINA_GRPC_RECV_BYTES'] = '0'

        self._s_task = self._pbar.add_task(
            ':arrow_up: Send', total=total, total_size=0, start=False
        )
        self._r_task = self._pbar.add_task(
            ':arrow_down: Recv', total=total, total_size=0, start=False
        )

    @staticmethod
    def _prepare_single_doc(d: 'Document'):
        if d.content_type in ('text', 'blob'):
            return d
        elif not d.blob and d.uri:
            d.load_uri_to_blob()
            return d
        elif d.tensor is not None:
            return d
        else:
            raise TypeError(f'unsupported input type {d!r} {d.content_type}')

    @staticmethod
    def _prepare_rank_doc(d: 'Document', _source: str = 'matches'):
        _get = lambda d: getattr(d, _source)
        if not _get(d):
            raise ValueError(f'`.rank()` requires every doc to have `.{_source}`')
        d = Client._prepare_single_doc(d)
        setattr(d, _source, [Client._prepare_single_doc(c) for c in _get(d)])
        return d

    def _iter_rank_docs(
        self, content, _source='matches'
    ) -> Generator['Document', None, None]:
        from rich import filesize
        from docarray import Document

        if hasattr(self, '_pbar'):
            self._pbar.start_task(self._s_task)

        for c in content:
            if isinstance(c, Document):
                yield self._prepare_rank_doc(c, _source)
            else:
                raise TypeError(f'unsupported input type {c!r}')

            if hasattr(self, '_pbar'):
                self._pbar.update(
                    self._s_task,
                    advance=1,
                    total_size=str(
                        filesize.decimal(
                            int(os.environ.get('JINA_GRPC_SEND_BYTES', '0'))
                        )
                    ),
                )

    def _get_rank_payload(self, content, kwargs):
        parameters = kwargs.get('parameters', {})
        model_name = parameters.get('model', '')
        payload = dict(
            on=f'/rank/{model_name}'.rstrip('/'),
            inputs=self._iter_rank_docs(
                content, _source=kwargs.get('source', 'matches')
            ),
            request_size=kwargs.get('batch_size', 8),
            total_docs=len(content) if hasattr(content, '__len__') else None,
        )
        if self._scheme == 'grpc' and self._authorization:
            payload.update(metadata=('authorization', self._authorization))
        elif self._scheme == 'http' and self._authorization:
            payload.update(headers={'Authorization': self._authorization})
        return payload

    def rank(self, docs: Iterable['Document'], **kwargs) -> 'DocumentArray':
        """Rank image-text matches according to the server CLIP model.
        Given a Document with nested matches, where the root is image/text and the matches is in another modality, i.e.
        text/image; this method ranks the matches according to the CLIP model.
        Each match now has a new score inside ``clip_score`` and matches are sorted descendingly according to this score.
        More details can be found in: https://github.com/openai/CLIP#usage
        :param docs: the input Documents
        :return: the ranked Documents in a DocumentArray.
        """
        self._prepare_streaming(
            not kwargs.get('show_progress'),
            total=len(docs),
        )
        results = DocumentArray()
        with self._pbar:
            self._client.post(
                **self._get_rank_payload(docs, kwargs),
                on_done=partial(self._gather_result, results=results),
            )
        return results

    async def arank(self, docs: Iterable['Document'], **kwargs) -> 'DocumentArray':
        from rich import filesize

        self._prepare_streaming(
            not kwargs.get('show_progress'),
            total=len(docs),
        )
        results = DocumentArray()
        async for da in self._async_client.post(**self._get_rank_payload(docs, kwargs)):
            if not results:
                self._pbar.start_task(self._r_task)
            results.extend(da)
            self._pbar.update(
                self._r_task,
                advance=len(da),
                total_size=str(
                    filesize.decimal(int(os.environ.get('JINA_GRPC_RECV_BYTES', '0')))
                ),
            )

        return results
