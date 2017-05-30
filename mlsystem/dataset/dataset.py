import mlsystem.db.data as db

class DictDataset(object):

    def __init__(self):
        self._labels = []
        self.used_features = ['numIVUsers','isLoopSimplifyForm','isEmpty','numIntToFloatCast','hasLoopPreheader','numTermBrBlocks','latchBlockTermOpcode']
        self._features = {k: [] for k in self.used_features}

    @property
    def features(self):
        return self._features

    @property
    def labels(self):
        return self._labels
    
    def append(self, label, features):
        assert(len(self.used_features) == len(self.features))
        for name,feature in zip(self.used_features, features):
            self._features[name].append(float(feature))
        self._labels.append(label)

    def size(self):
        return len(self._labels)

    def _get_label(self, loop_index, values):
        num_classes = len(values)
        label = int(values[loop_index])
        return label

class Dataset(object):
    def __init__(self, labels=[], features=[]):
        assert(len(labels) == len(features))
        self._labels = labels
        self._features = features

    @property
    def features(self):
        return self._features

    @property
    def labels(self):
        return self._labels
    
    def append(self, label, feature):
        self._features.append(feature)
        self._labels.append(label)

    def extend(self, labels, features):
        assert(len(labels) == len(features))
        self._labels.extend(labels)
        self._features.extend(features)

    def size(self):
        assert(len(self._labels) == len(self._features))
        return len(self._labels)

    def _get_label(self, loop_index, values):
        num_classes = len(values)
        label = [0] * num_classes
        label[int(values[loop_index])] = 1
        return label

    



class DatasetProducer(object):
    def __init__(self, run_ids, values, rule_parser, rule_stack, pass_name):
        db.init_database()
        self.runs = []
        for runs_set in run_ids:
            compared_runs = []
            for run_id in runs_set:
                compared_runs.append(db.Run.get(id=run_id, ignore=404))
            self.runs.append(compared_runs)

        self.values = values
        self.current_run_index = 0
        self.current_program_index = 0
        self.current_file_index = 0
        self.current_function_index = 0
        self.current_loop_index = 0
        self.rule_parser = rule_parser
        self.rule_stack = rule_stack
        self.pass_name = pass_name
        self.left_features = []
        self.left_label = None

    def _get_compare_to_loops(self, runs, function):
        run_loops = []
        for run in runs:
            run_loops.append({})
            s = db.Function.search().\
                                   query('match', run_id=run.meta.id).\
                                   query('match', function_name=function.function_name).\
                                   query('match', application=function.application).\
                                   query('match', filename=function.filename)
            response = s.execute()
            if response.success() and len(response):
                for function in response:
                    s = db.Loop.search().query('match', function_id=function.meta.id)
                    s = s[0:10000]
                    response_loops = s.execute()
                    for loop in response_loops:
                        run_loops[len(run_loops)-1][loop.loop_id] = loop 
        return run_loops

    def _get_features_vector(self, features_set):
        result_features = []
        properties = features_set.features_set._doc_type.mapping._d_.keys()
        for feature_property in ['numIVUsers','isLoopSimplifyForm','isEmpty','numIntToFloatCast','hasLoopPreheader','numTermBrBlocks','latchBlockTermOpcode']:
            if feature_property in properties:
                result_features.append(features_set.features_set[feature_property])
        return result_features

    def _get_features(self, loop):
        result_features = []
        s = db.LoopFeatures.search().query('match', block_id=loop.meta.id).sort('order')
        s = s[0:10000]
        loop_features = s.execute()
        before_pass = []
        before_all = []
        for features in loop_features:
            features_set = db.Features.get(id=features.features_id, ignore=404)
            if features_set:
                if features_set.place == 'Before':
                    if not before_all:
                        before_all = self._get_features_vector(features_set)
                    if features_set.pass_name == self.pass_name:
                        before_pass = self._get_features_vector(features_set)
                        result_features.append(before_pass)
        return result_features
    


    def next_batch(self, batch_size, dict_type=False):
        run_index = 0
        if dict_type:
            dataset = DictDataset()
        else:
            dataset = Dataset([], [])
        for run_set in self.runs:
            run = run_set[0]
            if run_index == self.current_run_index:
                s = db.ApplicationSearch(run_id=run.meta.id)
                response = s.execute()
                for program_index in range(self.current_program_index, len(response.facets.tags)):
                    application_name, _, _ = response.facets.tags[program_index]
                    s = db.FilenameSearch(application=application_name)
                    filenames_response = s.execute()
                    for file_index in range(self.current_file_index, len(filenames_response.facets.tags)):
                        filename, _, _ = filenames_response.facets.tags[file_index]
                        s = db.Function.search().query('match', run_id=run.meta.id).\
                                                 query('match', application=application_name).\
                                                 query('match', filename=filename)
                        s = s[0:10000]
                        functions = s.execute()
                        for function_index in range(self.current_function_index, len(functions)):
                            run_loops = self._get_compare_to_loops(run_set, functions[function_index])
                            result_loops = {}

                            if run_loops:
                                cur_run_loop = run_loops[0]
                                for loop_id, value in cur_run_loop.iteritems():
                                    is_found = True
                                    common_loops = []
                                    for loops in run_loops:
                                        if not loop_id in loops:
                                            is_found = False
                                        else:
                                            common_loops.append(loops[loop_id])
                                    if is_found:
                                        result_loops[loop_id] = common_loops
                            loop_index = 0
                            for loop_id, loops in result_loops.iteritems():
                                """if self.left_features:
                                    for features_inner in self.left_features:
                                        dataset.append(self.left_label, features_inner)"""
                                if loop_index == self.current_loop_index:
                                    # Find loop with best results.
                                    index_in_set = -1
                                    best_value = 0
                                    best_loop = None
                                    for i, loop in enumerate(loops):
                                        values_dict = {'exec_time': loop.exec_time,
                                                       'code_size': loop.code_size,
                                                       'llc_misses': loop.llc_misses}

                                        result_value = self.rule_parser.eval(self.rule_stack[:], values_dict)
                                        if not best_loop or self.rule_parser.better_value(result_value, best_value):
                                            index_in_set = i
                                            best_value = result_value
                                            best_loop = loop
                                    # Get class label for loop.
                                    label = dataset._get_label(index_in_set, self.values)
                                    # Get features for loop.
                                    features = self._get_features(best_loop)
                                    for features_inner in features:
                                        dataset.append(label, features_inner)
                                    self.current_loop_index += 1
                                    if dataset.size() >= batch_size:
                                        return dataset
                                        """self.left_features = dataset.features[batch_size:]
                                        self.left_label = label
                                        return Dataset(dataset.labels[:-(dataset.size() - batch_size)],
                                               dataset.features[:-(dataset.size() - batch_size)])"""
                                loop_index += 1
                            self.current_loop_index = 0
                            self.current_function_index += 1
                        self.current_function_index = 0
                        self.current_file_index += 1
                    self.current_file_index = 0
                    self.current_program_index += 1
                self.current_program_index = 0
                self.current_run_index += 1
            run_index += 1
        if dataset.size() == 0:
            self.current_run_index = 0
            self.current_program_index = 0
            self.current_file_index = 0
            self.current_function_index = 0
            self.current_loop_index = 0
        return dataset
