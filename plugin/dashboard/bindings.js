Vue.use(VueCharts);

const vm = new Vue({
    el: '#mrc-ui',
    data: {
        serverUrl: 'http://100.102.33.165:8081/bert_server',
        apiRoute: '/status/server',
        results: [],
        top_deck: [],
        second_deck: [],
        hist_num_request: {
            'last': -1,
            'value': [],
            'label': []
        },
        hist_num_client: {
            'last': -1,
            'value': [],
            'label': []
        },
        max_num_points: 720
    },
    mounted: function () {
        this.$nextTick(function () {
            this.refreshDatabase();
        })
    },
    computed: {
        databaseUrl: function () {
            return this.serverUrl + this.apiRoute
        },
        runningTime: function () {
            return moment(this.results.server_start_time).fromNow()
        }
    },
    methods: {
        histReqLabels: function (long) {
            return this.hist_num_request.label.slice(-(long ? this.max_num_points : 60))
        },
        histReqValues: function (long) {
            return this.hist_num_request.value.slice(-(long ? this.max_num_points : 60))
        },
        histClientLabels: function (long) {
            return this.hist_num_client.label.slice(-(long ? this.max_num_points : 60))
        },
        histClientValues: function (long) {
            return this.hist_num_client.value.slice(-(long ? this.max_num_points : 60))
        },
        refreshDatabase: function () {
            $.ajax({
                url: this.databaseUrl,
                dataType: 'text',
                cache: false,
                beforeSend: function () {
                    console.log("Loading");
                },
                error: function (jqXHR, textStatus, errorThrown) {
                    console.log(jqXHR);
                    console.log(textStatus);
                    console.log(errorThrown);
                },
                success: function (data) {
                    console.log('Success');
                    vm.results = JSON.parse(data);
                    vm.first_deck = [];
                    vm.second_deck = [];
                    vm.third_deck = [];
                    // add to top deck, high priority
                    vm.addToDeck('Data Req.', vm.results.statistic.num_data_request, vm.first_deck);
                    vm.addToDeck('Max RPS', (1 / vm.results.statistic.min_last_two_interval), vm.first_deck);
                    vm.addToDeck('Max Req./Client', vm.results.statistic.max_request_per_client, vm.first_deck);
                    vm.addToDeck('Num clients', vm.results.statistic.num_total_client, vm.first_deck);

                    // other dynamic stat to the second deck
                    vm.addToDeck('Sys Req.', vm.results.statistic.num_sys_request, vm.second_deck);
                    vm.addToDeck('Avg RPS', (1 / vm.results.statistic.avg_last_two_interval), vm.second_deck);
                    vm.addToDeck('Min Req./Client', vm.results.statistic.min_request_per_client, vm.second_deck);
                    vm.addToDeck('Avg Req./Client', vm.results.statistic.avg_request_per_client, vm.second_deck);

                    // mostly constant stat to the third deck
                    vm.addToDeck('Server version', vm.results.server_version, vm.second_deck);
                    vm.addToDeck('Running on', vm.results.cpu ? 'CPU' : 'GPU', vm.second_deck);
                    vm.addToDeck('Uptime', vm.runningTime, vm.second_deck);
                    vm.addToDeck('Workers', vm.results.num_worker, vm.second_deck);
                    vm.addToDeck('Max seq len', vm.results.max_seq_len, vm.second_deck);

                    vm.addNewTimeData(vm.hist_num_request, vm.results.statistic.num_data_request, true);
                    vm.addNewTimeData(vm.hist_num_client, vm.results.statistic.num_total_client, false);
                },
                complete: function () {
                    console.log('Finished all tasks');
                }
            });
        },
        addToDeck: function (text, value, deck, round) {
            round = typeof round !== 'undefined' ? round : true;
            round = (!isNaN(parseFloat(value)) && isFinite(value)) ? round : false;
            deck.push({'text': text, 'value': round ? Math.round(value) : value})
        },
        addNewTimeData: function (ds, new_val, delta) {
            if (ds.last >= 0)
                ds.value.push(new_val - (delta ? ds.last : 0));
            else
                ds.value.push(0);
            ds.last = new_val;
            ds.label.push(moment().format('h:mm:ss'));
            if (ds.label.length > vm.max_num_points) {
                ds.label = ds.label.slice(-vm.max_num_points);
                ds.value = ds.value.slice(-vm.max_num_points)
            }
        }
    }
});


setInterval(function () {
    vm.refreshDatabase();
    console.log('update database!')
}, 60 * 1000);
