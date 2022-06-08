package com.example.isenglish

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import com.example.isenglish.ml.LanguageClassification
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import org.json.JSONArray
import org.json.JSONObject
import java.io.InputStream

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val tv : TextView = findViewById(R.id.textView)

        val button : Button = findViewById(R.id.button)
        button.setOnClickListener {

            val ed: EditText = findViewById(R.id.inputIsEnglish)
            val inputIsEnglish: String = ed.text.toString()
            val seq = this.sequenize(this.splitLowerRemovePunctuation(inputIsEnglish))
            val floatseq = floatArrayOf(
                seq[0],
                seq[1],
                seq[2],
                seq[3],
                seq[4],
                seq[5],
                seq[6],
                seq[7],
                seq[8],
                seq[9],
                seq[10],
                seq[11],
                seq[12],
                seq[13],
                seq[14],
                seq[15],
                seq[16],
                seq[17],
                seq[18],
                seq[19],
                seq[20],
                seq[21],
                seq[22],
                seq[23],
                seq[24],
                seq[25],
                seq[26],
                seq[27],
                seq[28],
                seq[29],
                seq[30],
                seq[31]
            )


            val model = LanguageClassification.newInstance(this)

            val inputFeature = TensorBuffer.createFixedSize(intArrayOf(1, 32), DataType.FLOAT32)

            inputFeature.loadArray(floatseq)
            val outputs = model.process(inputFeature)
            val outputFeature = outputs.outputFeature0AsTensorBuffer.floatArray
            val english: String
            if (outputFeature[0] >= 0.5) {
                english = "Bahasa Indonesia"
            } else {
                english = "Bahasa Inggris"
            }

            val prob = outputFeature[0].toString()
            tv.text =
                "Hasil:$english\nProbability:$prob\nSequence:\n$seq\nInput\n$inputIsEnglish"
            model.close()
        }


    }
    fun wordIndex(): Map<String, *> {
        val json: String?
        val inputStream: InputStream = assets.open("IsEnglish.json")
        json = inputStream.bufferedReader().use { it.readText() }.toString()
        return JSONObject(json).toMap()
    }

    fun JSONObject.toMap(): Map<String, *> = keys().asSequence().associateWith { it ->
        when (val value = this[it])
        {
            is JSONArray ->
            {
                val map = (0 until value.length()).associate { Pair(it.toString(), value[it]) }
                JSONObject(map).toMap().values.toList()
            }
            is JSONObject -> value.toMap()
            JSONObject.NULL -> null
            else            -> value
        }
    }
    fun splitLowerRemovePunctuation(source:String): Array<String> {
        val str = source.replace("[!\"#$%&'()*+,-./:;<=>?@\\[\\]^_`{|}~]".toRegex(), "").trim()
        return str.split("\\s+".toRegex()).toTypedArray()
    }
    fun sequenize(source: Array<String>):ArrayList<Float> {
        var counter = 32
        val seq = arrayListOf<Float>()
        for (i in source){
            if(i in wordIndex().keys){
                if (counter == 0){
                    return seq
                }else{
                    counter -= 1
                    seq.add(wordIndex().get(i).toString().toFloat())
                }
            }else{
                if (counter == 0){
                    return seq
                }else{
                    counter -= 1
                    seq.add(1f)
                }
            }
        }
        for (i in 1..counter){
            seq.add(0f)
        };return seq
    }
}



