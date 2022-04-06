import builtins
import pickle


class MyPickler(pickle.Prickler):
    def persistent_id(self, obj):
        # Instead of pickling MemoRecord as a regular class instance, we emit a
        # persistent ID.
        if isinstance(obj, MemoRecord):
            # Here, our persistent ID is simply a tuple, containing a tag and a
            # key, which refers to a specific record in the database.
            return ("MemoRecord", obj.key)
        else:
            # If obj does not have a persistent ID, return None. This means obj
            # needs to be pickled as usual.
            return None


safe_builtins = {
    'pymooCFD',
    'range',
    'complex',
    'set',
    'frozenset',
    'slice'
}


class MyUnpickler(pickle.Unpickler):

     def persistent_load(self, pid):
        # This method is invoked whenever a persistent ID is encountered.
        # Here, pid is the tuple returned by DBPickler.
        cursor = self.connection.cursor()
        type_tag, key_id = pid
        if type_tag == "MemoRecord":
            # Fetch the referenced record from the database and return it.
            cursor.execute("SELECT * FROM memos WHERE key=?", (str(key_id),))
            key, task = cursor.fetchone()
            return MemoRecord(key, task)
        else:
            # Always raises an error if you cannot return the correct object.
            # Otherwise, the unpickler will think None is the object referenced
            # by the persistent ID.
            raise pickle.UnpicklingError("unsupported persistent object")

    def find_class(self, module, name):
        # Only allow safe classes from builtins.
        if module == "builtins" and name in safe_builtins:
            return getattr(builtins, name)
        # Forbid everything else.
        raise pickle.UnpicklingError("global '%s.%s' is forbidden" %
                                     (module, name))


# some object to persist
# usually, one would have some store or bookkeeping in place
bar = (1, 2)


# The create/load implementation of the persistent id
# extends pickling/unpickling
class PersistentPickler(pickle.Pickler):
    def persistent_id(self, obj):
        """Return a persistent id for the `bar` object only"""
        return "it's a bar" if obj is bar else None


class PersistentUnpickler(pickle.Unpickler):
    def persistent_load(self, pers_id):
        """Return the object identified by the persistent id"""
        if pers_id == "it's a bar":
           return bar
        raise pickle.UnpicklingError("This is just an example for one persistent object!")


# we can now dump and load the persistent object
foo = {'bar': bar}
with open("foo.pkl", "wb") as out_stream:
    PersistentPickler(out_stream).dump(foo)

with open("foo.pkl", "rb") as in_stream:
    foo2 = PersistentUnpickler(in_stream).load()

assert foo2 is not foo     # regular objects are not persistent
assert foo2['bar'] is bar  # persistent object identity is preserved
