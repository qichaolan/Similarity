/*
 * $Id$ Copyright (C) 2009 Klaus Reimer <k@ailis.de>
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
 * associated documentation files (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge, publish, distribute,
 * sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all copies or
 * substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
 * NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

package com.weibo.yar.codec;

import de.ailis.pherialize.Mixed;
import de.ailis.pherialize.MixedArray;
import de.ailis.pherialize.Unserializer;
import de.ailis.pherialize.exceptions.UnserializeException;

import java.util.ArrayList;
import java.util.List;


/**
 * Unserializes a PHP serialize format string into a Java object.
 *
 * @author Klaus Reimer (k@ailis.de)
 * @version $Revision$
 */

public class PHPUnserializer extends Unserializer {
    /**
     * The current pointer in the data
     */
    private int pos;

    /**
     * The data to unserialize
     */
    private final String data;

    /**
     * The object history for resolving references
     */
    private final List<Object> history;


    /**
     * Constructor
     *
     * @param data The data to unserialize
     */

    public PHPUnserializer(final String data) {
        super(data);
        this.data = data;
        this.pos = 0;
        this.history = new ArrayList<Object>();
    }


    /**
     * Unserializes the next object in the data stream.
     *
     * @return The unserializes object
     */

    @Override
    public Mixed unserializeObject() {
        char type;
        Mixed result;

        type = this.data.charAt(this.pos);
        switch (type) {
            case 's':
                result = unserializeString();
                break;

            case 'i':
                result = unserializeLong();
                break;

            case 'd':
                result = unserializeDouble();
                break;

            case 'b':
                result = unserializeBoolean();
                break;

            case 'N':
                result = unserializeNull();
                break;

            case 'a':
                return unserializeArray();

            case 'R':
                result = unserializeReference();
                break;

            default:
                throw new UnserializeException("Unable to unserialize unknown type " + type, this.pos);
        }

        this.history.add(result);
        return result;
    }


    /**
     * Unserializes the next object in the data stream into a String.
     *
     * @return The unserialized String
     */

    private Mixed unserializeString() {
        int pos, length;

        pos = this.data.indexOf(':', this.pos + 2);
        length = Integer.parseInt(this.data.substring(this.pos + 2, pos));
        this.pos = pos + length + 4;
        return new Mixed(this.data.substring(pos + 2, pos + 2 + length));
    }


    /**
     * Unserializes the next object in the data stream into an Integer.
     *
     * @return The unserialized Integer
     */

    private Mixed unserializeLong() {
        Long result;
        int pos;

        pos = this.data.indexOf(';', this.pos + 2);
        result = Long.valueOf(this.data.substring(this.pos + 2, pos));
        this.pos = pos + 1;
        return new Mixed(result);
    }


    /**
     * Unserializes the next object in the data stream into an Double.
     *
     * @return The unserialized Double
     */

    private Mixed unserializeDouble() {
        Double result;
        int pos;

        pos = this.data.indexOf(';', this.pos + 2);
        result = Double.valueOf(this.data.substring(this.pos + 2, pos));
        this.pos = pos + 1;
        return new Mixed(result);
    }


    /**
     * Unserializes the next object in the data stream as a reference.
     *
     * @return The unserialized reference
     */

    private Mixed unserializeReference() {
        int index;
        int pos;

        pos = this.data.indexOf(';', this.pos + 2);
        index = Integer.parseInt(this.data.substring(this.pos + 2, pos));
        this.pos = pos + 1;
        return (Mixed) this.history.get(index - 1);
    }


    /**
     * Unserializes the next object in the data stream into a Boolean.
     *
     * @return The unserialized Boolean
     */

    private Mixed unserializeBoolean() {
        Boolean result;

        result = Boolean.valueOf(this.data.charAt(this.pos + 2) == '1');
        this.pos += 4;
        return new Mixed(result);
    }


    /**
     * Unserializes the next object in the data stream into a Null
     *
     * @return The unserialized Null
     */

    private Mixed unserializeNull() {
        this.pos += 2;
        return null;
    }


    /**
     * Unserializes the next object in the data stream into an array. This method returns an
     * ArrayList if the unserialized array has numerical keys starting with 0 or a HashMap
     * otherwise.
     *
     * @return The unserialized array
     */

    private Mixed unserializeArray() {
        Mixed result;
        MixedArray array;
        int pos;
        int max;
        int i;
        Object key, value;

        pos = this.data.indexOf(':', this.pos + 2);
        max = Integer.parseInt(this.data.substring(this.pos + 2, pos));
        this.pos = pos + 2;
        array = new MixedArray(max);
        result = new Mixed(array);
        this.history.add(result);
        for (i = 0; i < max; i++) {
            key = unserializeObject();
            this.history.remove(this.history.size() - 1);
            value = unserializeObject();
            array.put(key, value);
        }
        this.pos++;
        return result;
    }
}
