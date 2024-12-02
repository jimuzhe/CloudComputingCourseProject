import java.io.IOException;
import java.util.*;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

public class SegmentedWordCount {
    public static class TokenizerMapper extends Mapper<Object, Text, Text, IntWritable> {
        private final static IntWritable one = new IntWritable(1);
        private Text word = new Text();

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            // 直接处理每行的词
            String w = value.toString().trim();
            if (!w.isEmpty()) {
                word.set(w);
                context.write(word, one);
            }
        }
    }

    public static class IntSumReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
        private TreeMap<String, Integer> wordCountMap;
        private final int TOP_N = 1000;
        @Override
        protected void setup(Context context) {
            wordCountMap = new TreeMap<>();
        }
        public void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            wordCountMap.put(key.toString(), sum);
        }
        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            // 按词频排序
            List<Map.Entry<String, Integer>> sortedEntries = new ArrayList<>(wordCountMap.entrySet());
            sortedEntries.sort((e1, e2) -> e2.getValue().compareTo(e1.getValue()));
            // 输出TOP N结果
            int count = 0;
            for (Map.Entry<String, Integer> entry : sortedEntries) {
                if (count >= TOP_N)
                    break;
                context.write(new Text(entry.getKey()), new IntWritable(entry.getValue()));
                count++;
            }
        }
    }
    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
        if (otherArgs.length != 2) {
            System.err.println("Usage: wordcount <in> <out>");
            System.exit(2);
        }

        Job job = Job.getInstance(conf, "segmented word count");
        job.setJarByClass(SegmentedWordCount.class);
        job.setMapperClass(TokenizerMapper.class);
        job.setReducerClass(IntSumReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
        job.setNumReduceTasks(1); // 确保输出文件只有一个

        FileInputFormat.addInputPath(job, new Path(otherArgs[0]));
        FileOutputFormat.setOutputPath(job, new Path(otherArgs[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}